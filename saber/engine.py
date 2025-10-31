"""
SABER Engine Implementation
"""
import re
import time
import duckdb
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

import lotus
from lotus.models import SentenceTransformersRM, LM
from lotus.vector_store import FaissVS 

from .config import LOTUS_DEFAULT_RM_MODEL
from .llm_config import get_default_llm_config
from .backends import LOTUSBackend, DocETLBackend, PalimpzestBackend
from .core import QueryRewriter, SQLParser, SemanticSetOperations, SemRewriter
from .utils import quote_dot_columns, get_column_context
from .query_generator import SABERQueryGenerator
from .benchmark import BenchmarkTracker, BenchmarkStats, extract_lotus_stats

logger = logging.getLogger(__name__)

class SaberEngine:
    """
    SABER Engine with vectorized operations supporting multiple backends.
    
    Semantic Functions:
    1. SEM_WHERE: Semantic filtering
    2. SEM_SELECT: Semantic projection
    3. SEM_EXCEPT: Semantic difference
    3-1. SEM_EXCEPT_ALL: Semantic difference (including duplicates)
    4. SEM_INTERSECT: Semantic intersection
    4-1. SEM_INTERSECT_ALL: Semantic intersection (including duplicates)
    5. SEM_JOIN: Semantic join matching
    6. SEM_GROUP_BY: Semantic grouping
    7. SEM_AGG: Semantic aggregation
    8. SEM_DISTINCT: Semantic deduplication
    9. SEM_ORDER_BY: Semantic ordering
    """
    
    def __init__(self, backend: str = 'lotus', openai_api_key: str = None, use_ast_rewriter: bool = True, use_local_llm: bool = False, enable_benchmarking: bool = False):
        """
        Initialize SABER Engine.
        
        Args:
            backend: Default backend for query rewriting ('lotus', 'docetl', 'palimpzest')
            openai_api_key: OpenAI API key. If None and use_local_llm=False, checks OPENAI_API_KEY env var.
            use_ast_rewriter: Use AST-based rewriting (recommended) vs regex-based (legacy)
            use_local_llm: Force use of local VLLM instead of OpenAI. Default is False.
            enable_benchmarking: Enable automatic benchmarking for query() and query_ast() calls. Default is False.
        """
        # Initialize DuckDB connection and dataframe storage
        self.conn = duckdb.connect()
        self._dataframes = {}
        
        # Set default backend for query rewriting
        self.default_backend = backend
        self.use_ast_rewriter = use_ast_rewriter
        self.enable_benchmarking = enable_benchmarking
        
        # Initialize LLM configuration
        self.llm_config = get_default_llm_config(openai_api_key, use_local=use_local_llm)
        
        # Get API key (may be None for local VLLM mode)
        api_key = self.llm_config.api_key
        
        # Initialize LOTUS settings
        lotus_config = self.llm_config.get_model_config('lotus')
        self.lm = LM(model=lotus_config['model'], api_key=lotus_config['api_key'])
        self.rm = SentenceTransformersRM(model=LOTUS_DEFAULT_RM_MODEL)
        self.vs = FaissVS()
        lotus.settings.configure(lm=self.lm, rm=self.rm, vs=self.vs)
        
        # Initialize backends - engine supports all three
        lotus_config = self.llm_config.get_model_config('lotus')
        docetl_config = self.llm_config.get_model_config('docetl')
        palimpzest_config = self.llm_config.get_model_config('palimpzest')
        
        self.backends = {
            'lotus': LOTUSBackend(
                api_key=lotus_config['api_key'],
                model=lotus_config['model'],
                api_base=lotus_config['api_base']
            ),
            'docetl': DocETLBackend(
                api_key=docetl_config['api_key'],
                model=docetl_config['model'],
                api_base=docetl_config['api_base']
            ), 
            'palimpzest': PalimpzestBackend(
                api_key=palimpzest_config['api_key'],
                model=palimpzest_config['model'],
                api_base=palimpzest_config['api_base']
            )
        }
        
        # Pass LM reference to LOTUS backend for cost tracking
        self.backends['lotus']._lm = self.lm
        
        # Initialize helper modules
        query_rewriter_config = self.llm_config.get_model_config('query_rewriter')
        self.query_rewriter = QueryRewriter(
            query_rewriter_config['model'], 
            query_rewriter_config['api_base'],
            query_rewriter_config['api_key']
        )
        self.sql_parser = SQLParser()
        
        self.semantic_ops = SemanticSetOperations(self.backends)
        
        if use_ast_rewriter:
            self.ast_rewriter = SemRewriter(dialect='duckdb')
        
        # Initialize query generator
        qg_config = self.llm_config.get_model_config('query_rewriter')
        self.query_generator = SABERQueryGenerator(
            model=qg_config['model'],
            api_key=qg_config['api_key'],
            api_base=qg_config['api_base'],
            backend=backend  # Use engine's default backend
        )
        
        # Initialize benchmark tracker
        self.benchmark = BenchmarkTracker()
        self._accumulated_cost = 0.0
        self._accumulated_latency = 0.0
        self._accumulated_sql_time = 0.0
        self._accumulated_rewriting_time = 0.0
    
    def _accumulate_backend_stats(self, backend):
        """Accumulate cost and semantic execution time from backend's last operation."""
        if not self.enable_benchmarking:
            return
        if hasattr(backend, 'last_stats') and backend.last_stats:
            self._accumulated_cost += backend.last_stats.total_cost
            self._accumulated_latency += backend.last_stats.total_semantic_execution_time_seconds
            logger.debug(f"Operation stats - Cost: ${backend.last_stats.total_cost:.6f}, "
                        f"Semantic Time: {backend.last_stats.total_semantic_execution_time_seconds:.2f}s")
    
    def _track_sql_execution(self, func, *args, **kwargs):
        """Track SQL execution time for non-semantic operations."""
        if not self.enable_benchmarking:
            return func(*args, **kwargs)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        self._accumulated_sql_time += elapsed_time
        logger.debug(f"SQL execution time: {elapsed_time:.2f}s")
        return result
    
    def _track_rewriting_time(self, func, *args, **kwargs):
        """Track time spent rewriting backend-free queries."""
        if not self.enable_benchmarking:
            return func(*args, **kwargs)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        self._accumulated_rewriting_time += elapsed_time
        logger.debug(f"Rewriting time: {elapsed_time:.2f}s")
        return result
    
    def rewrite_prompt(self, user_prompt: str, column_context: str, aliases: Dict[str, str], backend: str, operation: str = None) -> str:
        """Delegate to query rewriter."""
        return self.query_rewriter.rewrite_prompt(user_prompt, column_context, aliases, backend, operation)
    
    def register_table(self, name: str, df: pd.DataFrame):
        """Register table for vectorized processing."""
        self._dataframes[name] = df.copy()
        self.conn.register(name, df)
    
    def get_backend(self, backend_name: str):
        """Get a specific backend instance."""
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not supported. Available: {list(self.backends.keys())}")
        return self.backends[backend_name]
    
    def update_api_key(self, new_api_key: str, use_local: bool = False):
        """
        Update the API key for all components.
        
        Args:
            new_api_key: New OpenAI API key (can be None to switch to local)
            use_local: Whether to use local VLLM
        """
        # Update LLM configuration
        self.llm_config.update_api_key(new_api_key, use_local=use_local)
        
        # Update LOTUS configuration
        lotus_config = self.llm_config.get_model_config('lotus')
        self.lm = LM(model=lotus_config['model'], api_key=lotus_config['api_key'])
        self.rm = SentenceTransformersRM(model=LOTUS_DEFAULT_RM_MODEL)
        self.vs = FaissVS()
        lotus.settings.configure(lm=self.lm, rm=self.rm, vs=self.vs)
        
        # Update backends with new configuration
        for backend_name, backend in self.backends.items():
            config = self.llm_config.get_model_config(backend_name)
            backend.set_model_config(
                model=config['model'],
                api_base=config['api_base'],
                api_key=config['api_key']
            )
        
        # Update query rewriter
        query_rewriter_config = self.llm_config.get_model_config('query_rewriter')
        self.query_rewriter = QueryRewriter(
            query_rewriter_config['model'],
            query_rewriter_config['api_base'],
            query_rewriter_config['api_key']
        )
        
        # Update query generator
        self.query_generator = SABERQueryGenerator(
            model=query_rewriter_config['model'],
            api_key=query_rewriter_config['api_key'],
            api_base=query_rewriter_config['api_base'],
            backend=self.default_backend
        )
    
    def generate(
        self,
        question: str,
        tables: List[str] = None,
        schema: str = None,
        examples: List[Dict] = None,
        backend: str = None,
        max_sample_rows: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 512
    ) -> str:
        """
        Generate SABER SQL query from natural language question.
        
        Args:
            question: Natural language question
            tables: List of table names to include (required unless schema provided)
            schema: Optional schema string (alternative to using registered tables)
            examples: Optional custom few-shot examples
            backend: Backend for semantic operations ('lotus' or 'docetl', default: engine's backend)
            max_sample_rows: Number of sample rows to include per table (default: 5)
            temperature: LLM temperature for generation
            max_tokens: Maximum tokens to generate (default: 512)
            
        Returns:
            Generated SABER SQL query
            
        Example:
            engine.register_table('employees', df)
            query = engine.generate("Which engineers have ML experience?", tables=['employees'])
            result = engine.query(query)
        """
        # Build schema from registered tables if not provided
        if schema is None:
            if tables is None:
                tables = list(self._dataframes.keys())
            
            df_dict = {name: self._dataframes[name] for name in tables if name in self._dataframes}
            
            if not df_dict:
                raise ValueError("No tables registered. Use register_table() first or provide schema.")
            
            return self.query_generator.generate(
                question=question,
                tables=tables,
                df=df_dict,
                examples=examples,
                backend=backend,
                max_sample_rows=max_sample_rows,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            return self.query_generator.generate(
                question=question,
                tables=tables,
                schema=schema,
                examples=examples,
                backend=backend,
                max_sample_rows=max_sample_rows,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    def _evaluate_fragment(self, sql_fragment: str) -> pd.DataFrame:
        """Evaluate a SQL fragment and return the resulting DataFrame."""
        frag = sql_fragment.strip()

        # unwrap one pair of outer parentheses, if present
        if frag.startswith("(") and frag.endswith(")"):
            frag = frag[1:-1].strip()

        # unwrap one pair of outer quotes, if present
        if len(frag) >= 2 and frag[0] == frag[-1] and frag[0] in {"'", '"'}:
            frag = frag[1:-1]

        # decide how to execute / fetch
        if frag.lower().startswith("select"):
            return self.query(frag)
        if frag in self._dataframes:
            return self._dataframes[frag]
        raise ValueError(f"Unknown fragment {frag!r}")
    
    def _handle_regular_joins(self, sql: str) -> tuple[str, Optional[pd.DataFrame]]:
        """Handle regular SQL JOINs by creating a temporary joined table with prefixed columns."""
        join_info = self.sql_parser.extract_regular_join_info(sql)
        if not join_info:
            return sql, None
        
        left_table = join_info['left_table']
        left_alias = join_info['left_alias']
        right_table = join_info['right_table']
        right_alias = join_info['right_alias']
        join_condition = join_info['join_condition']
        
        logging.info(f"Detected JOIN: {left_table} ({left_alias}) JOIN {right_table} ({right_alias}) ON {join_condition}")
        
        if left_table not in self._dataframes or right_table not in self._dataframes:
            raise ValueError(f"Tables '{left_table}' or '{right_table}' not found in registered dataframes.")
        
        left_df = self._dataframes[left_table].copy()
        right_df = self._dataframes[right_table].copy()
        
        # Add prefixes to column names to avoid conflicts
        left_df.columns = [f"{left_alias}.{col}" for col in left_df.columns]
        right_df.columns = [f"{right_alias}.{col}" for col in right_df.columns]
        
        # Parse join condition
        join_parts = join_condition.split('=')
        if len(join_parts) != 2:
            raise ValueError(f"Unsupported join condition format: {join_condition}")
        
        left_join_col = join_parts[0].strip()
        right_join_col = join_parts[1].strip()
        
        # Ensure join columns have proper prefixes
        if '.' not in left_join_col:
            left_join_col = f"{left_alias}.{left_join_col}"
        if '.' not in right_join_col:
            right_join_col = f"{right_alias}.{right_join_col}"
        
        # Perform the join using pandas
        try:
            joined_df = left_df.merge(right_df, left_on=left_join_col, right_on=right_join_col, how='inner')
            temp_table_name = f"temp_join_{left_table}_{right_table}"
            
            # Modify SQL to use the temporary table
            # Replace the entire FROM...JOIN...ON clause with the temp table
            modified_sql = re.sub(
                r'FROM\s+\w+(?:\s+AS\s+\w+)?\s+JOIN\s+\w+(?:\s+AS\s+\w+)?\s+ON\s+[^)]+?(?=\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|\s*$)',
                f'FROM {temp_table_name}',
                sql,
                flags=re.IGNORECASE
            )
            
            return modified_sql, joined_df
        except Exception as e:
            logging.error(f"Error performing join: {e}")
            return sql, None
    
    def _handle_intersect_calls(self, sql: str, matches: list[tuple[tuple[int, int], str, str]], is_set: bool) -> str:
        """Evaluate every SEM_INTERSECT(_ALL) occurrence and replace with a temp view name registered in DuckDB."""
        offset = 0
        for idx, (span, q1, q2) in enumerate(matches, 1):
            df1 = self._evaluate_fragment(q1)
            df2 = self._evaluate_fragment(q2)
            intersect_df = self.semantic_ops.intersect_operation(df1, df2, self.rm, is_set=is_set)

            view_name = f"temp_intersect_{'set' if is_set else 'all'}_{idx}"
            self.conn.register(view_name, intersect_df)

            start, end = span
            sql = sql[:start + offset] + view_name + sql[end + offset:]
            offset += len(view_name) - (end - start)

        return sql

    def _handle_except_calls(self, sql: str, matches: list[tuple[tuple[int, int], str, str]], is_set: bool) -> str:
        """Evaluate every SEM_EXCEPT(_ALL) occurrence and replace with a temp view name registered in DuckDB."""
        offset = 0
        for idx, (span, q1, q2) in enumerate(matches, 1):
            df1 = self._evaluate_fragment(q1)
            df2 = self._evaluate_fragment(q2)
            diff_df = self.semantic_ops.except_operation(df1, df2, self.rm, is_set=is_set)

            view_name = f"temp_except_{'set' if is_set else 'all'}_{idx}"
            self.conn.register(view_name, diff_df)

            start, end = span
            sql = sql[:start + offset] + view_name + sql[end + offset:]
            offset += len(view_name) - (end - start)

        return sql
    
    def _handle_intersect_except_calls(self, sql: str) -> str:
        """Handle SEM_INTERSECT and SEM_EXCEPT operations."""
        # Extract all intersect/except operations
        intersect_except_ops = self.sql_parser.extract_intersect_except_operations(sql)

        sql = self._handle_intersect_calls(sql, intersect_except_ops['intersect_all'], is_set=False)
        sql = self._handle_intersect_calls(sql, intersect_except_ops['intersect_set'], is_set=True)
        sql = self._handle_except_calls(sql, intersect_except_ops['except_all'], is_set=False)
        sql = self._handle_except_calls(sql, intersect_except_ops['except_set'], is_set=True)

        return sql
    
    def _get_join_column_context(self, join_info: Dict[str, str]) -> str:
        """
        Generate column context for queries with JOINs, showing prefixed column names.
        
        Args:
            join_info: Dictionary with JOIN information
            
        Returns:
            Schema description string with prefixed columns
        """
        left_table = join_info['left_table']
        left_alias = join_info['left_alias']
        right_table = join_info['right_table']
        right_alias = join_info['right_alias']
        
        context = "Schema Information:\n"
        context += f"After JOIN, available columns will be:\n"
        
        # Add left table columns with prefix
        if left_table in self._dataframes:
            left_df = self._dataframes[left_table]
            context += f"From table '{left_table}' (alias '{left_alias}'):\n"
            for col in left_df.columns:
                context += f"  - {left_alias}.{col}\n"
        
        # Add right table columns with prefix
        if right_table in self._dataframes:
            right_df = self._dataframes[right_table]
            context += f"From table '{right_table}' (alias '{right_alias}'):\n"
            for col in right_df.columns:
                context += f"  - {right_alias}.{col}\n"
                
        return context
    
    def query_ast(self, sql: str, explain: bool = False) -> pd.DataFrame:
        if not self.use_ast_rewriter:
            raise RuntimeError("AST rewriter not enabled. Set use_ast_rewriter=True in constructor.")
        
        # Extract aliases for query rewriting context
        aliases = self.sql_parser.extract_table_aliases(sql)
        
        # Patterns for backend-free semantic operations
        # SEM_WHERE, SEM_SELECT: simple prompt-only format
        simple_pattern = r"(SEM_(?:WHERE|SELECT))\s*\(\s*'([^']*)'\s*\)"
        # SEM_ORDER_BY: with column argument for unified queries: SEM_ORDER_BY(column, 'prompt')
        order_with_col_pattern = r"(SEM_ORDER_BY)\s*\(\s*([^,']+)\s*,\s*'([^']*)'\s*\)"
        # SEM_ORDER_BY: prompt-only (for LOTUS backend-specific queries)
        order_no_col_pattern = r"(SEM_ORDER_BY)\s*\(\s*'([^']*)'\s*\)"
        # SEM_AGG: with column argument: SEM_AGG(column, 'prompt') AS alias
        agg_with_col_pattern = r"(SEM_AGG)\s*\(\s*([^,']+)\s*,\s*'([^']*)'\s*\)\s+AS\s+(\w+)"
        # SEM_AGG: prompt-only format: SEM_AGG('prompt') AS alias
        agg_no_col_pattern = r"(SEM_AGG)\s*\(\s*'([^']*)'\s*\)\s+AS\s+(\w+)"
        # SEM_JOIN: two tables, prompt
        join_pattern = r"(SEM_JOIN)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*'([^']*)'\s*\)"
        
        # Check if we have any backend-free operations
        has_backend_free = (bool(re.search(simple_pattern, sql)) or 
                           bool(re.search(order_with_col_pattern, sql)) or
                           bool(re.search(order_no_col_pattern, sql)) or
                           bool(re.search(agg_with_col_pattern, sql)) or
                           bool(re.search(agg_no_col_pattern, sql)) or
                           bool(re.search(join_pattern, sql)))
        
        # Handle backend-free queries by rewriting them
        if has_backend_free:
            rewriting_start = time.time() if self.enable_benchmarking else None
            
            # Check if we have JOINs to determine the correct column context
            join_info = self.sql_parser.extract_regular_join_info(sql)
            if join_info:
                # Generate column context based on JOIN structure
                column_context = self._get_join_column_context(join_info)
            else:
                column_context = get_column_context(self._dataframes)
            
            # Rewrite SEM_WHERE and SEM_SELECT
            def rewrite_simple_match(match):
                func_name = match.group(1)
                user_prompt = match.group(2)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(simple_pattern, rewrite_simple_match, sql)
            
            # Rewrite SEM_ORDER_BY with column
            def rewrite_order_with_col_match(match):
                func_name = match.group(1)
                column = match.group(2).strip()
                user_prompt = match.group(3)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                
                # For LOTUS, embed column in prompt; for DocETL/Palimpzest, keep as argument
                if self.default_backend == 'lotus':
                    # Don't include column as separate arg for LOTUS
                    return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}')"
                else:
                    # DocETL and Palimpzest need column as first arg
                    return f"{func_name}({column}, '{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(order_with_col_pattern, rewrite_order_with_col_match, sql)
            
            # Rewrite SEM_ORDER_BY without column (LOTUS-specific)
            def rewrite_order_no_col_match(match):
                func_name = match.group(1)
                user_prompt = match.group(2)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(order_no_col_pattern, rewrite_order_no_col_match, sql)
            
            # Rewrite SEM_AGG with column
            def rewrite_agg_with_col_match(match):
                func_name = match.group(1)
                column = match.group(2).strip()
                user_prompt = match.group(3)
                alias = match.group(4)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                
                # For LOTUS, embed column in prompt; for DocETL, keep as argument
                if self.default_backend == 'lotus':
                    return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}') AS {alias}"
                else:
                    # DocETL needs column as first arg
                    return f"{func_name}({column}, '{rewritten_prompt_escaped}', '{self.default_backend}') AS {alias}"
            
            sql = re.sub(agg_with_col_pattern, rewrite_agg_with_col_match, sql)
            
            # Rewrite SEM_AGG without column
            def rewrite_agg_no_col_match(match):
                func_name = match.group(1)
                user_prompt = match.group(2)
                alias = match.group(3)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}') AS {alias}"
            
            sql = re.sub(agg_no_col_pattern, rewrite_agg_no_col_match, sql)
            
            # Rewrite SEM_JOIN
            def rewrite_join_match(match):
                func_name = match.group(1)
                left_table = match.group(2).strip()
                right_table = match.group(3).strip()
                user_prompt = match.group(4)
                
                # Generate JOIN-specific column context
                if left_table in self._dataframes and right_table in self._dataframes:
                    left_df = self._dataframes[left_table]
                    right_df = self._dataframes[right_table]
                    join_context = f"Left table ({left_table}) columns: {', '.join(left_df.columns)}\n"
                    join_context += f"Right table ({right_table}) columns: {', '.join(right_df.columns)}"
                else:
                    join_context = column_context
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, join_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}({left_table}, {right_table}, '{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(join_pattern, rewrite_join_match, sql)
            logging.info(f"Rewritten SQL for backend '{self.default_backend}':\n{sql}\n")
            
            # Track rewriting time
            if rewriting_start is not None:
                self._accumulated_rewriting_time += time.time() - rewriting_start
        
        try:
            canonical_sql, operations = self.ast_rewriter.rewrite(sql, optimize_plan=True)
            
            if explain:
                logging.info("=== SABER Query Plan ===")
                logging.info(self.ast_rewriter.explain())
                logging.info("=== Canonical SQL ===")
                logging.info(canonical_sql)
                logging.info("=== Operations ===")
                logging.info(operations)

            result = self._execute_with_semantic_ops(canonical_sql, operations)
            return result
            
        except Exception as e:
            logging.error(f"Error in AST execution: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _execute_with_semantic_ops(self, canonical_sql: str, operations: List) -> pd.DataFrame:
        # Check if the final SELECT contains a JOIN
        final_select_pattern = r'SELECT\s+.*?\s+FROM\s+.*?\s+JOIN\s+'
        has_final_join = bool(re.search(final_select_pattern, canonical_sql, re.IGNORECASE | re.DOTALL))
        
        cte_pattern = r'WITH\s+_child\s+AS\s+\((.*?)\)'
        match = re.search(cte_pattern, canonical_sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            base_sql = match.group(1)
            
            # Check if we have set operations (JOIN, INTERSECT, EXCEPT) - if so, skip base SQL execution
            has_set_operation = any(op.op_type in ('join', 'intersect', 'intersect_all', 'except', 'except_all') 
                                   for op in operations)
            
            # Extract the table name from base_sql to get all columns
            # Pattern: FROM table_name or FROM table_name AS alias
            table_match = re.search(r'FROM\s+([a-zA-Z_][\w]*)', base_sql, re.IGNORECASE)
            base_table_name = table_match.group(1) if table_match else None
            
            # If the final SELECT has a JOIN and we have semantic operations, we need to perform JOIN first
            if has_final_join and not has_set_operation:
                # Extract the JOIN from the final SELECT
                join_match = re.search(r'FROM\s+(\w+)\s+JOIN\s+(\w+)\s+AS\s+(\w+)\s+ON\s+([^\s]+)\s*=\s*([^\s]+)', 
                                      canonical_sql, re.IGNORECASE)
                if join_match:
                    # Get the left table name from base_sql
                    left_table_match = re.search(r'FROM\s+(\w+)\s+AS\s+(\w+)', base_sql, re.IGNORECASE)
                    if left_table_match:
                        left_table_name = left_table_match.group(1)
                        left_alias = left_table_match.group(2)
                        
                        right_table_name = join_match.group(2)
                        right_alias = join_match.group(3)
                        left_join_col_raw = join_match.group(4).strip()
                        right_join_col_raw = join_match.group(5).strip()
                        
                        if left_table_name in self._dataframes and right_table_name in self._dataframes:
                            # Get the full dataframes (not the optimized base_sql result)
                            left_df = self._dataframes[left_table_name].copy()
                            right_df = self._dataframes[right_table_name].copy()
                            
                            # Add prefixes to column names
                            left_df.columns = [f"{left_alias}.{col}" for col in left_df.columns]
                            right_df.columns = [f"{right_alias}.{col}" for col in right_df.columns]
                            
                            # Parse join column names (remove table aliases if present)
                            left_join_col = left_join_col_raw.split('.')[-1]
                            right_join_col = right_join_col_raw.split('.')[-1]
                            
                            # Add prefixes for the actual join
                            left_join_col = f"{left_alias}.{left_join_col}"
                            right_join_col = f"{right_alias}.{right_join_col}"
                            
                            # Perform the join
                            current_df = left_df.merge(right_df, left_on=left_join_col, right_on=right_join_col, how='inner')
                        else:
                            # Fall back to SQL execution
                            base_result = self._track_sql_execution(lambda: self.conn.execute(base_sql).fetch_df())
                            current_df = base_result
                    else:
                        # Couldn't parse base_sql, fall back
                        base_result = self._track_sql_execution(lambda: self.conn.execute(base_sql).fetch_df())
                        current_df = base_result
                else:
                    # Couldn't parse JOIN, fall back to normal execution
                    base_result = self._track_sql_execution(lambda: self.conn.execute(base_sql).fetch_df())
                    current_df = base_result
            elif not has_set_operation:
                # No JOIN in final SELECT, execute base SQL normally
                # But if we have semantic operations, get ALL columns from the base table
                # so that columns referenced in semantic prompts are available
                if operations and base_table_name and base_table_name in self._dataframes:
                    # Get all columns from the registered table
                    current_df = self._dataframes[base_table_name].copy()
                    # Apply any WHERE clauses from base_sql if present
                    where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|LIMIT|$)', base_sql, re.IGNORECASE | re.DOTALL)
                    if where_match:
                        # Execute the base SQL to get filtered rows, but select all columns
                        where_clause = where_match.group(1).strip()
                        # Construct a query that selects all columns with the WHERE clause
                        full_query = f"SELECT * FROM {base_table_name} WHERE {where_clause}"
                        try:
                            current_df = self._track_sql_execution(lambda: self.conn.execute(full_query).fetch_df())
                        except:
                            # If WHERE clause fails, just execute the original base_sql
                            current_df = self._track_sql_execution(lambda: self.conn.execute(base_sql).fetch_df())
                else:
                    base_result = self._track_sql_execution(lambda: self.conn.execute(base_sql).fetch_df())
                    current_df = base_result
            else:
                # For set operations, we'll start with None and the operation will populate it
                current_df = None
            
            # Track if we've done a GROUP BY to inform SEM_AGG
            grouped_by_column = None
            
            # Extract GROUP BY columns from the final SELECT query for SEM_AGG
            group_by_columns = []
            final_query_preview = canonical_sql.split('\n')[-1]
            group_by_match = re.search(r'GROUP\s+BY\s+([^\s]+(?:\s*,\s*[^\s]+)*)', final_query_preview, re.IGNORECASE)
            if group_by_match:
                # Extract column names from GROUP BY clause
                group_by_str = group_by_match.group(1).strip()
                group_by_columns = [col.strip() for col in group_by_str.split(',')]
            
            for op in operations:
                # Backend selection depends on operation type
                # For most operations: args[0] = prompt, args[1] = backend, args[2+] = optional
                # For SEM_JOIN: args[0] = left_table, args[1] = right_table, args[2] = prompt, args[3] = backend
                # For SEM_GROUP_BY: args[0] = column, args[1] = k, args[2] = backend
                # For SEM_AGG: args = [prompt, backend, alias] OR [column, prompt, backend, alias]
                # For SEM_ORDER_BY: args = [prompt, backend] OR [column, prompt, backend]
                if op.op_type == 'join':
                    backend_name = op.args[3].strip("'") if len(op.args) > 3 else 'lotus'
                elif op.op_type == 'groupby':
                    backend_name = op.args[2].strip("'") if len(op.args) > 2 else 'lotus'
                elif op.op_type == 'agg':
                    # AGG can have 3 args (prompt, backend, alias) or 4 args (column, prompt, backend, alias)
                    backend_name = op.args[2].strip("'") if len(op.args) == 4 else (op.args[1].strip("'") if len(op.args) >= 2 else 'lotus')
                elif op.op_type == 'orderby':
                    # ORDER_BY can have 2 args (prompt, backend) or 3 args (column, prompt, backend)
                    backend_name = op.args[2].strip("'") if len(op.args) == 3 else (op.args[1].strip("'") if len(op.args) >= 2 else 'lotus')
                else:
                    backend_name = op.args[1].strip("'") if len(op.args) > 1 else 'lotus'
                
                backend_impl = self.backends.get(backend_name)
                if not backend_impl:
                    backend_impl = self.backends['lotus']
                
                if op.op_type == 'where':
                    prompt = op.args[0].strip("'") if op.args else ''
                    current_df = backend_impl.sem_where(current_df, prompt)
                    self._accumulate_backend_stats(backend_impl)
                    
                elif op.op_type == 'select':
                    prompt = op.args[0].strip("'") if op.args else ''
                    alias = op.args[2].strip("'") if len(op.args) > 2 else 'extracted'
                    current_df = backend_impl.sem_select(current_df, prompt, alias)
                    self._accumulate_backend_stats(backend_impl)
                    
                elif op.op_type == 'join':
                    # SEM_JOIN needs two tables
                    if len(op.args) >= 3:
                        left_table = op.args[0].strip("'\"")
                        right_table = op.args[1].strip("'\"")
                        prompt = op.args[2].strip("'\"")
                        
                        # Get the dataframes
                        if left_table in self._dataframes and right_table in self._dataframes:
                            df1 = self._dataframes[left_table]
                            df2 = self._dataframes[right_table]
                            current_df = backend_impl.sem_join(df1, df2, prompt, left_table, right_table)
                            self._accumulate_backend_stats(backend_impl)
                        else:
                            raise ValueError(f"Tables '{left_table}' or '{right_table}' not found in registered dataframes.")
                
                elif op.op_type == 'intersect':
                    # SEM_INTERSECT - set semantics (no duplicates)
                    if len(op.args) >= 2:
                        query1 = op.args[0].strip("'\"")
                        query2 = op.args[1].strip("'\"")
                        df1 = self._evaluate_fragment(query1)
                        df2 = self._evaluate_fragment(query2)
                        current_df = self.semantic_ops.intersect_operation(df1, df2, self.rm, is_set=True)
                
                elif op.op_type == 'intersect_all':
                    # SEM_INTERSECT_ALL - bag semantics (with duplicates)
                    if len(op.args) >= 2:
                        query1 = op.args[0].strip("'\"")
                        query2 = op.args[1].strip("'\"")
                        df1 = self._evaluate_fragment(query1)
                        df2 = self._evaluate_fragment(query2)
                        current_df = self.semantic_ops.intersect_operation(df1, df2, self.rm, is_set=False)
                
                elif op.op_type == 'except':
                    # SEM_EXCEPT - set semantics (no duplicates)
                    if len(op.args) >= 2:
                        query1 = op.args[0].strip("'\"")
                        query2 = op.args[1].strip("'\"")
                        df1 = self._evaluate_fragment(query1)
                        df2 = self._evaluate_fragment(query2)
                        current_df = self.semantic_ops.except_operation(df1, df2, self.rm, is_set=True)
                
                elif op.op_type == 'except_all':
                    # SEM_EXCEPT_ALL - bag semantics (with duplicates)
                    if len(op.args) >= 2:
                        query1 = op.args[0].strip("'\"")
                        query2 = op.args[1].strip("'\"")
                        df1 = self._evaluate_fragment(query1)
                        df2 = self._evaluate_fragment(query2)
                        current_df = self.semantic_ops.except_operation(df1, df2, self.rm, is_set=False)
                    
                    
                elif op.op_type == 'groupby':
                    text_col = op.args[0].strip("'") if op.args else '*'
                    k = int(op.args[1]) if len(op.args) > 1 else 8
                    current_df = backend_impl.sem_group_by(current_df, text_col, k)
                    self._accumulate_backend_stats(backend_impl)
                    # Track that we've grouped - SEM_AGG will need to group by cluster_id
                    grouped_by_column = 'cluster_id'
                    
                elif op.op_type == 'orderby':
                    # SEM_ORDER_BY has two formats:
                    # 2 args: [prompt, backend] - prompt-only (LOTUS style)
                    # 3 args: [column, prompt, backend] - column-specific (DocETL/Palimpzest style)
                    column = None
                    if len(op.args) == 3:
                        # Column-specific: column, prompt, backend
                        column = op.args[0].strip("'")
                        prompt = op.args[1].strip("'")
                    else:
                        # Prompt-only: prompt, backend
                        prompt = op.args[0].strip("'") if op.args else ''
                    current_df = backend_impl.sem_order_by(current_df, prompt, column)
                    self._accumulate_backend_stats(backend_impl)
                    
                elif op.op_type == 'distinct':
                    text_col = op.args[0].strip("'") if op.args else '*'
                    alias = op.args[-1].strip("'") if len(op.args) > 2 else None
                    result_df = backend_impl.sem_distinct(current_df, text_col)
                    self._accumulate_backend_stats(backend_impl)
                    # Rename the deduped column to the alias if specified
                    if alias and text_col in result_df.columns:
                        current_df = result_df.rename(columns={text_col: alias})
                    else:
                        current_df = result_df
                    
                elif op.op_type == 'agg':
                    # SEM_AGG has two formats:
                    # 3 args: [prompt, backend, alias] - prompt-only (LOTUS style)
                    # 4 args: [column, prompt, backend, alias] - column-specific (DocETL style)
                    column = None
                    if len(op.args) == 4:
                        # Column-specific: column, prompt, backend, alias
                        column = op.args[0].strip("'")
                        prompt = op.args[1].strip("'")
                        alias = op.args[3].strip("'")
                    else:
                        # Prompt-only: prompt, backend, alias
                        prompt = op.args[0].strip("'") if op.args else ''
                        alias = op.args[2].strip("'") if len(op.args) > 2 else 'agg_result'
                    
                    # Use GROUP BY columns from final query if available, otherwise use grouped_by_column from SEM_GROUP_BY
                    if group_by_columns:
                        group_by_col = group_by_columns
                    elif grouped_by_column:
                        group_by_col = [grouped_by_column]
                    else:
                        group_by_col = None
                    
                    current_df = backend_impl.sem_agg(current_df, prompt, alias, group_by_col, column)
                    self._accumulate_backend_stats(backend_impl)
            
            # Register the result as final temp table and execute the final SELECT
            self.conn.register('_sem_final', current_df)
            
            # Track SEM_AGG aliases for wrapping in ANY_VALUE() if needed
            sem_agg_aliases = [op.args[3].strip("'") if len(op.args) == 4 else op.args[2].strip("'") 
                              for op in operations if op.op_type == 'agg']
            
            # Extract and execute the final SELECT query
            final_select_pattern = r'SELECT\s+.*?\s+FROM\s+_sem_\d+(.*?)$'
            final_match = re.search(final_select_pattern, canonical_sql, re.IGNORECASE | re.DOTALL)
            if final_match:
                # Replace ALL _sem_N references with _sem_final (not just in FROM clause)
                final_query = canonical_sql.split('\n')[-1]
                final_query = re.sub(r'_sem_\d+', '_sem_final', final_query)
                
                # If the final query has GROUP BY, wrap SEM_AGG columns in ANY_VALUE()
                if re.search(r'\bGROUP\s+BY\b', final_query, re.IGNORECASE):
                    for alias in sem_agg_aliases:
                        # Replace bare alias with ANY_VALUE(alias) in SELECT clause
                        # Match: SELECT ..., alias, ... or SELECT alias, ...
                        # Don't match if already wrapped (e.g., ANY_VALUE(alias))
                        pattern = r'\b' + re.escape(alias) + r'\b(?!\s*\))'  # Not followed by )
                        # Only replace in SELECT clause (before FROM)
                        select_part, _, rest = final_query.partition(' FROM ')
                        if select_part:
                            select_part = re.sub(pattern, f'ANY_VALUE({alias})', select_part)
                            final_query = select_part + ' FROM ' + rest
                
                # If we performed a JOIN upfront, remove the JOIN clause from final query
                if has_final_join:
                    # Check if we have a semantic JOIN operation
                    has_sem_join = any(op.op_type == 'join' for op in operations)
                    
                    if has_sem_join:
                        # After SEM_JOIN, columns are prefixed (e.g., products.category)
                        # Need to fix the JOIN ON condition to use the correct prefixed column
                        # Pattern: ON sections.name = _sem_final.category
                        # Should become: ON sections.name = _sem_final."products.category"
                        
                        # Find the JOIN ON condition
                        join_pattern = r'JOIN\s+(\w+)\s+(?:AS\s+(\w+)\s+)?ON\s+([^\s]+)\s*=\s*_sem_final\.(\w+)'
                        join_match = re.search(join_pattern, final_query, re.IGNORECASE)
                        
                        if join_match:
                            join_table = join_match.group(1)
                            join_condition_left = join_match.group(3)
                            referenced_column = join_match.group(4)
                            
                            # Find the actual prefixed column name in current_df
                            # It could be like "products.category" or "left.category"
                            matching_cols = [col for col in current_df.columns if col.endswith(f'.{referenced_column}')]
                            
                            if matching_cols:
                                # Use the first matching column (should only be one)
                                actual_column = matching_cols[0]
                                
                                # Replace the JOIN ON condition with the actual column name
                                # Don't quote here - let quote_dot_columns handle it later
                                final_query = re.sub(
                                    r'(JOIN\s+' + join_table + r'\s+(?:AS\s+\w+\s+)?ON\s+' + re.escape(join_condition_left) + r'\s*=\s*)_sem_final\.\w+',
                                    r'\1_sem_final.' + actual_column,
                                    final_query,
                                    flags=re.IGNORECASE
                                )
                    else:
                        # Regular case: remove the JOIN clause entirely (was handled by upfront JOIN)
                        final_query = re.sub(
                            r'FROM\s+_sem_final\s+JOIN\s+\w+\s+AS\s+\w+\s+ON\s+[^\s]+\s*=\s*[^\s]+',
                            'FROM _sem_final',
                            final_query,
                            flags=re.IGNORECASE
                        )
                
                # Quote columns with dots for proper SQL execution
                final_query = quote_dot_columns(final_query, current_df.columns.tolist())
                result = self._track_sql_execution(lambda: self.conn.execute(final_query).fetch_df())
                return result
            else:
                return current_df
        else:
            return self._track_sql_execution(lambda: self.conn.execute(canonical_sql).fetch_df())
    
    def explain(self, sql: str) -> str:
        if not self.use_ast_rewriter:
            return "EXPLAIN not available. Enable AST rewriter (use_ast_rewriter=True)."
        
        try:
            canonical_sql, operations = self.ast_rewriter.rewrite(sql, optimize_plan=True)
            
            lines = []
            lines.append("=== SABER Query Plan ===\n")
            lines.append(self.ast_rewriter.explain())
            
            lines.append("\n=== Canonical SQL ===")
            lines.append(canonical_sql)
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL with automatic vectorization and backend support."""
        
        # Extract aliases for query rewriting context
        aliases = self.sql_parser.extract_table_aliases(sql)
        
        # Patterns for backend-free semantic operations
        # SEM_WHERE, SEM_SELECT: simple prompt-only format
        simple_pattern = r"(SEM_(?:WHERE|SELECT))\s*\(\s*'([^']*)'\s*\)"
        # SEM_ORDER_BY: with column argument for unified queries: SEM_ORDER_BY(column, 'prompt')
        order_with_col_pattern = r"(SEM_ORDER_BY)\s*\(\s*([^,']+)\s*,\s*'([^']*)'\s*\)"
        # SEM_ORDER_BY: prompt-only (for LOTUS backend-specific queries)
        order_no_col_pattern = r"(SEM_ORDER_BY)\s*\(\s*'([^']*)'\s*\)"
        # SEM_AGG: with column argument: SEM_AGG(column, 'prompt') AS alias
        agg_with_col_pattern = r"(SEM_AGG)\s*\(\s*([^,']+)\s*,\s*'([^']*)'\s*\)\s+AS\s+(\w+)"
        # SEM_AGG: prompt-only format: SEM_AGG('prompt') AS alias
        agg_no_col_pattern = r"(SEM_AGG)\s*\(\s*'([^']*)'\s*\)\s+AS\s+(\w+)"
        # SEM_JOIN: two tables, prompt
        join_pattern = r"(SEM_JOIN)\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*'([^']*)'\s*\)"
        
        # Check if we have any backend-free operations
        has_backend_free = (bool(re.search(simple_pattern, sql)) or 
                           bool(re.search(order_with_col_pattern, sql)) or
                           bool(re.search(order_no_col_pattern, sql)) or
                           bool(re.search(agg_with_col_pattern, sql)) or
                           bool(re.search(agg_no_col_pattern, sql)) or
                           bool(re.search(join_pattern, sql)))
        
        # Handle backend-free queries by rewriting them
        if has_backend_free:
            # Start timing for rewriting
            rewriting_start = time.time() if self.enable_benchmarking else None
            
            # Check if we have JOINs to determine the correct column context
            join_info = self.sql_parser.extract_regular_join_info(sql)
            if join_info:
                # Generate column context based on JOIN structure
                column_context = self._get_join_column_context(join_info)
            else:
                column_context = get_column_context(self._dataframes)
            
            # Rewrite SEM_WHERE and SEM_SELECT
            def rewrite_simple_match(match):
                func_name = match.group(1)
                user_prompt = match.group(2)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(simple_pattern, rewrite_simple_match, sql)
            
            # Rewrite SEM_ORDER_BY with column
            def rewrite_order_with_col_match(match):
                func_name = match.group(1)
                column = match.group(2).strip()
                user_prompt = match.group(3)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                
                # For LOTUS, embed column in prompt; for DocETL/Palimpzest, keep as argument
                if self.default_backend == 'lotus':
                    # Don't include column as separate arg for LOTUS
                    return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}')"
                else:
                    # DocETL and Palimpzest need column as first arg
                    return f"{func_name}({column}, '{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(order_with_col_pattern, rewrite_order_with_col_match, sql)
            
            # Rewrite SEM_ORDER_BY without column (LOTUS-specific)
            def rewrite_order_no_col_match(match):
                func_name = match.group(1)
                user_prompt = match.group(2)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(order_no_col_pattern, rewrite_order_no_col_match, sql)
            
            # Rewrite SEM_AGG with column
            def rewrite_agg_with_col_match(match):
                func_name = match.group(1)
                column = match.group(2).strip()
                user_prompt = match.group(3)
                alias = match.group(4)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                
                # For LOTUS, embed column in prompt; for DocETL, keep as argument
                if self.default_backend == 'lotus':
                    return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}') AS {alias}"
                else:
                    # DocETL needs column as first arg
                    return f"{func_name}({column}, '{rewritten_prompt_escaped}', '{self.default_backend}') AS {alias}"
            
            sql = re.sub(agg_with_col_pattern, rewrite_agg_with_col_match, sql)
            
            # Rewrite SEM_AGG without column
            def rewrite_agg_no_col_match(match):
                func_name = match.group(1)
                user_prompt = match.group(2)
                alias = match.group(3)
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, column_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}('{rewritten_prompt_escaped}', '{self.default_backend}') AS {alias}"
            
            sql = re.sub(agg_no_col_pattern, rewrite_agg_no_col_match, sql)
            
            # Rewrite SEM_JOIN
            def rewrite_join_match(match):
                func_name = match.group(1)
                left_table = match.group(2).strip()
                right_table = match.group(3).strip()
                user_prompt = match.group(4)
                
                # Generate JOIN-specific column context
                if left_table in self._dataframes and right_table in self._dataframes:
                    left_df = self._dataframes[left_table]
                    right_df = self._dataframes[right_table]
                    join_context = f"Left table ({left_table}) columns: {', '.join(left_df.columns)}\n"
                    join_context += f"Right table ({right_table}) columns: {', '.join(right_df.columns)}"
                else:
                    join_context = column_context
                
                rewritten_prompt = self.rewrite_prompt(
                    user_prompt, join_context, aliases, self.default_backend, operation=func_name
                )
                rewritten_prompt_escaped = rewritten_prompt.replace("'", "''")
                return f"{func_name}({left_table}, {right_table}, '{rewritten_prompt_escaped}', '{self.default_backend}')"
            
            sql = re.sub(join_pattern, rewrite_join_match, sql)
            logging.info(f"Rewritten SQL for backend '{self.default_backend}':\n{sql}\n")
            
            # Track rewriting time
            if rewriting_start is not None:
                self._accumulated_rewriting_time += time.time() - rewriting_start
        
        # Handle regular JOINs first
        modified_sql, joined_df = self._handle_regular_joins(sql)
        temp_table_name = None
        
        if joined_df is not None:
            temp_table_name = f"temp_join_table_{len(self._dataframes)}"
            # Extract the actual temp table name from the modified SQL
            temp_match = re.search(r'FROM\s+(\w+)', modified_sql, re.IGNORECASE)
            if temp_match:
                temp_table_name = temp_match.group(1)
            
            self.conn.register(temp_table_name, joined_df)
            self._dataframes[temp_table_name] = joined_df
            sql = modified_sql

        # Handle INTERSECT/EXCEPT operations
        sql = self._handle_intersect_except_calls(sql)

        # Extract semantic operations
        semantic_ops = self.sql_parser.extract_semantic_operations(sql)
        table_matches = self.sql_parser.extract_table_matches(sql)
        
        logging.info(f"Table matches found: {table_matches}")
        
        # Check if we have semantic operations to process
        has_semantic_ops = any(semantic_ops.values())
        
        if has_semantic_ops and table_matches and (table_matches[0] == 'SEM_JOIN' or (table_matches[0] in self._dataframes)):
            table_name = table_matches[0]
            if table_name != 'SEM_JOIN':
                df = self._dataframes[table_name].copy()
            
            # Apply semantic operations in order
            
            # 1. Semantic JOINs
            for left_table, right_table, user_prompt, system in semantic_ops['join']:
                if left_table in self._dataframes and right_table in self._dataframes:
                    backend = self.get_backend(system)
                    df1 = self._dataframes[left_table]
                    df2 = self._dataframes[right_table]
                    df = backend.sem_join(df1, df2, user_prompt, left_table, right_table)
                    self._accumulate_backend_stats(backend)
                else:
                    raise ValueError(f"Tables '{left_table}' or '{right_table}' not found in registered dataframes.")
            
            # 2. Semantic WHERE clauses
            for user_prompt, system in semantic_ops['where']:
                backend = self.get_backend(system)
                df = backend.sem_where(df, user_prompt)
                self._accumulate_backend_stats(backend)
            
            # 3. Semantic GROUP BY
            for column, number_of_groups, system in semantic_ops['group']:
                backend = self.get_backend(system)
                df = backend.sem_group_by(df, column, number_of_groups)
                self._accumulate_backend_stats(backend)
            
            # 4. Semantic aggregations
            for column, user_prompt, system, alias in semantic_ops['agg']:
                backend = self.get_backend(system)
                # If group_matches is present, use the grouped column
                # This assumes that semantic grouping is applied in only one column
                group_by_col = ["cluster_id"] if semantic_ops['group'] else None
                df = backend.sem_agg(df, user_prompt, alias, group_by_col, column)
                self._accumulate_backend_stats(backend)
            
            # 5. Semantic SELECT
            for user_prompt, system, alias in semantic_ops['select']:
                backend = self.get_backend(system)
                df = backend.sem_select(df, user_prompt, alias)
                self._accumulate_backend_stats(backend)
            
            # 6. Semantic DISTINCT
            for column, system in semantic_ops['distinct']:
                backend = self.get_backend(system)
                df = backend.sem_distinct(df, column)
                self._accumulate_backend_stats(backend)
            
            # 7. Semantic ORDER BY
            for column, user_prompt, system in semantic_ops['order']:
                backend = self.get_backend(system)
                df = backend.sem_order_by(df, user_prompt, column)
                self._accumulate_backend_stats(backend)
            
            # If we have ordering, add ordering column
            if semantic_ops['order']:
                df = df.reset_index(drop=True)
                df['_semantic_order'] = range(len(df))
            
            # Register processed data and modify SQL
            temp_table = f"{table_name}_vectorized"
            self.conn.register(temp_table, df)

            # Store column names for rollback after processing
            column_names_rollback = [] 

            # Replace semantic functions with processed columns
            if semantic_ops['join']:
                logging.info(f"Original SQL: {sql}")
                # Replace SEM_JOIN(...) with just the temp_table name (no arguments) in the SQL
                sem_join_pattern = r"SEM_JOIN\s*\(\s*[^)]+\)"
                modified_sql = re.sub(sem_join_pattern, f"{temp_table}", sql, flags=re.IGNORECASE)
                logging.info(f"Modified SQL after SEM_JOIN: {modified_sql}")
            else:
                modified_sql = sql.replace(f"FROM {table_name}", f"FROM {temp_table}")
            
            if semantic_ops['where']:
                modified_sql = re.sub(self.sql_parser.patterns['where'], "TRUE", modified_sql)
            
            if semantic_ops['group']:
                for column, number_of_groups, system in semantic_ops['group']:
                    if system.lower() == 'lotus':
                        modified_sql = re.sub(self.sql_parser.patterns['group'], "cluster_id", modified_sql)
                    elif system.lower() == 'docetl':
                        modified_sql = re.sub(self.sql_parser.patterns['group'], "cluster_name", modified_sql)
                    else:
                        raise NotImplementedError(f"Semantic GROUP BY not implemented for backend '{system}'")
            
            if semantic_ops['agg']:
                for _, user_prompt, system, alias in semantic_ops['agg']:
                    if system.lower() == 'lotus' or system.lower() == 'docetl':
                        replacement = f"ANY_VALUE({alias})"
                        column_names_rollback.append([alias, replacement])
                    else:
                        raise NotImplementedError(f"Semantic AGG not implemented for backend '{system}'")
                    modified_sql = re.sub(self.sql_parser.patterns['agg'], replacement, modified_sql)
            
            if semantic_ops['select']:
                for user_prompt, system, alias in semantic_ops['select']:
                    if system.lower() == 'lotus':
                        column = alias if alias else '_map'
                    if "GROUP BY" in modified_sql.upper():
                        replacement = f"ANY_VALUE({column})"
                    else:
                        replacement = f"{alias}"
                    
                    # Build a specific pattern for this exact SEM_SELECT call
                    # Escape special regex characters in the user_prompt
                    escaped_prompt = re.escape(user_prompt)
                    escaped_system = re.escape(system)
                    escaped_alias = re.escape(alias)
                    
                    specific_pattern = rf"SEM_SELECT\s*\(\s*'{escaped_prompt}'\s*,\s*'{escaped_system}'\s*\)\s+AS\s+{escaped_alias}"
                    modified_sql = re.sub(specific_pattern, replacement, modified_sql, flags=re.IGNORECASE)
            
            if semantic_ops['distinct']:
                modified_sql = re.sub(self.sql_parser.patterns['distinct'], r"\1", modified_sql)
            
            if semantic_ops['order']:
                for column, user_prompt, system in semantic_ops['order']:
                    sem_order_pattern = r'SEM_ORDER_BY\s*\([^)]+\)'
                    if system.lower() == 'lotus':
                        modified_sql = re.sub(sem_order_pattern, "_semantic_order", modified_sql, flags=re.IGNORECASE)
                    elif system.lower() == 'docetl':
                        modified_sql = re.sub(sem_order_pattern, "_rank", modified_sql, flags=re.IGNORECASE)
                    elif system.lower() == 'palimpzest':
                        modified_sql = re.sub(sem_order_pattern, "_relevant", modified_sql, flags=re.IGNORECASE)
                    else:
                        raise NotImplementedError(f"Semantic ORDER BY not implemented for backend '{system}'")

            # Quote columns with dots and replace table name
            modified_sql = quote_dot_columns(modified_sql, df)
            logging.info(f"Final modified SQL for execution:\n{modified_sql}\n")

            result = self._track_sql_execution(lambda: self.conn.sql(modified_sql).df())

            # Remove internal columns from result if they exist
            internal_columns = ['_semantic_order', '_rank', '_relevant']
            columns_to_drop = [col for col in internal_columns if col in result.columns]
            if columns_to_drop:
                result = result.drop(columns=columns_to_drop)
            
            # Restore original column names
            for original, replacement in column_names_rollback:
                # case-insensitive replacement
                result.columns = [re.sub(re.escape(replacement), original, col, flags=re.IGNORECASE) for col in result.columns]
            
            # Cleanup
            try:
                self.conn.execute(f"DROP VIEW IF EXISTS {temp_table}")
            except:
                pass

            # At the end, cleanup temporary table
            if temp_table_name:
                try:
                    self.conn.execute(f"DROP VIEW IF EXISTS {temp_table_name}")
                    if temp_table_name in self._dataframes:
                        del self._dataframes[temp_table_name]
                except:
                    pass
            
            return result

        # Clean up temporary table
        if temp_table_name:
            try:
                self.conn.execute(f"DROP VIEW IF EXISTS {temp_table_name}")
                if temp_table_name in self._dataframes:
                    del self._dataframes[temp_table_name]
            except:
                pass
        
        # Execute final SQL
        return self._track_sql_execution(lambda: self.conn.sql(sql).df())
    
    def reset_benchmark(self):
        """Reset benchmark statistics."""
        self.benchmark.reset()
        self._accumulated_cost = 0.0
        self._accumulated_latency = 0.0
        self._accumulated_sql_time = 0.0
        self._accumulated_rewriting_time = 0.0
        # Reset LOTUS LM stats
        if hasattr(self, 'lm') and hasattr(self.lm, 'reset_stats'):
            self.lm.reset_stats()
        # Reset backend last_stats
        for backend in self.backends.values():
            if hasattr(backend, 'last_stats'):
                backend.last_stats = BenchmarkStats()
    
    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Get current benchmark statistics."""
        return self.benchmark.get_summary()
    
    def query_with_benchmark(self, sql: str, reset: bool = True) -> tuple[pd.DataFrame, BenchmarkStats]:
        """
        Execute query and return results with benchmark statistics.
        
        Args:
            sql: SQL query to execute
            reset: Reset stats before execution (default: True)
            
        Returns:
            Tuple of (result DataFrame, BenchmarkStats with cost and latency)
        """
        # Temporarily enable benchmarking for this call
        original_state = self.enable_benchmarking
        self.enable_benchmarking = True
        
        try:
            if reset:
                self.reset_benchmark()
            
            start_time = time.time()
            result = self.query(sql)
            elapsed_time = time.time() - start_time
            
            # Return accumulated cost and measured latency
            stats = BenchmarkStats(
                total_cost=self._accumulated_cost,
                total_execution_time_seconds=elapsed_time,
                total_semantic_execution_time_seconds=self._accumulated_latency,
                total_non_semantic_execution_time_seconds=self._accumulated_sql_time,
                total_rewriting_time_seconds=self._accumulated_rewriting_time
            )
            
            return result, stats
        finally:
            # Restore original benchmarking state
            self.enable_benchmarking = original_state
    
    def query_ast_with_benchmark(self, sql: str, reset: bool = True, explain: bool = False) -> tuple[pd.DataFrame, BenchmarkStats]:
        """
        Execute query with AST rewriter and return results with benchmark statistics.
        
        Args:
            sql: SQL query to execute
            reset: Reset stats before execution (default: True)
            explain: Whether to print explanation (default: False)
            
        Returns:
            Tuple of (result DataFrame, BenchmarkStats with cost and latency)
        """
        # Temporarily enable benchmarking for this call
        original_state = self.enable_benchmarking
        self.enable_benchmarking = True
        
        try:
            if reset:
                self.reset_benchmark()
            
            start_time = time.time()
            result = self.query_ast(sql, explain=explain)
            elapsed_time = time.time() - start_time
            
            # Return accumulated cost and latency from individual operations
            stats = BenchmarkStats(
                total_execution_time_seconds=elapsed_time,
                total_cost=self._accumulated_cost,
                total_semantic_execution_time_seconds=self._accumulated_latency,
                total_non_semantic_execution_time_seconds=self._accumulated_sql_time,
                total_rewriting_time_seconds=self._accumulated_rewriting_time
            )
            
            return result, stats
        finally:
            # Restore original benchmarking state
            self.enable_benchmarking = original_state

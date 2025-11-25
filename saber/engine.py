"""
SABER Engine Implementation
"""
import re
import time
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

import lotus
from lotus.models import SentenceTransformersRM, LM, LiteLLMRM
from lotus.vector_store import FaissVS 

from .config import LOTUS_DEFAULT_RM_MODEL, LOCAL_EMBEDDING_MODEL
from .llm_config import get_default_llm_config
from .backends import LOTUSBackend, DocETLBackend, PalimpzestBackend
from .core import QueryRewriter, SQLParser, SemanticSetOperations, SemRewriter
from .utils import quote_dot_columns, get_column_context, DatabaseAdapter
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
    
    def __init__(self, backend: str = 'lotus', openai_api_key: str = None, use_ast_rewriter: bool = True, use_local_llm: bool = False, enable_benchmarking: bool = False, fallback_enabled: bool = True, max_query_retries: int = 3, db_type: str = 'duckdb', **db_kwargs):
        """
        Initialize SABER Engine.
        
        Args:
            backend: Default backend for query rewriting ('lotus', 'docetl', 'palimpzest')
            openai_api_key: OpenAI API key. If None and use_local_llm=False, checks OPENAI_API_KEY env var.
            use_ast_rewriter: Use AST-based rewriting (recommended) vs regex-based (legacy)
            use_local_llm: Force use of local VLLM instead of OpenAI. Default is False.
            enable_benchmarking: Enable automatic benchmarking for query() and query_ast() calls. Default is False.
            fallback_enabled: Enable fallback to simple queries when LLM generation fails. Default is True.
            max_query_retries: Maximum retry attempts for query generation. Default is 3.
            db_type: Database type ('duckdb', 'sqlite', 'mysql'). Default is 'duckdb'.
            **db_kwargs: Additional database connection parameters (e.g., host, user, password for MySQL)
        """
        # Initialize database connection and dataframe storage
        self.db_adapter = DatabaseAdapter(db_type, **db_kwargs)
        self.conn = self.db_adapter.conn
        self.db_type = db_type.lower()
        self._dataframes = {}
        
        # Set default backend for query rewriting
        self.default_backend = backend
        self.use_ast_rewriter = use_ast_rewriter
        self.enable_benchmarking = enable_benchmarking
        
        # Initialize LLM configuration
        self.llm_config = get_default_llm_config(openai_api_key, use_local=use_local_llm)
        
        # Initialize LOTUS settings
        lotus_config = self.llm_config.get_model_config('lotus')
        self.lm = LM(model=lotus_config['model'], api_key=lotus_config['api_key'])
        if use_local_llm:
            if LOCAL_EMBEDDING_MODEL.startswith("litellm_proxy"):
                self.rm = LiteLLMRM(model=LOCAL_EMBEDDING_MODEL)
            else:
                self.rm = SentenceTransformersRM(model=LOCAL_EMBEDDING_MODEL)
        else:
            # self.rm = SentenceTransformersRM(model=LOTUS_DEFAULT_RM_MODEL)
            self.rm = LiteLLMRM(model=LOTUS_DEFAULT_RM_MODEL)
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
                api_base=lotus_config['api_base'],
                embedding_model=lotus_config.get('embedding_model'),
                embedding_api_base=lotus_config.get('embedding_api_base')
            ),
            'docetl': DocETLBackend(
                api_key=docetl_config['api_key'],
                model=docetl_config['model'],
                api_base=docetl_config['api_base'],
                embedding_model=docetl_config.get('embedding_model'),
                embedding_api_base=docetl_config.get('embedding_api_base')
            ), 
            'palimpzest': PalimpzestBackend(
                api_key=palimpzest_config['api_key'],
                model=palimpzest_config['model'],
                api_base=palimpzest_config['api_base'],
                embedding_model=palimpzest_config.get('embedding_model'),
                embedding_api_base=palimpzest_config.get('embedding_api_base')
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
            self.ast_rewriter = SemRewriter(dialect=self.db_type)
        
        # Initialize query generator
        qg_config = self.llm_config.get_model_config('query_rewriter')
        self.query_generator = SABERQueryGenerator(
            model=qg_config['model'],
            api_key=qg_config['api_key'],
            api_base=qg_config['api_base'],
            backend=backend,
            max_retries=max_query_retries,
            fallback_enabled=fallback_enabled,
            db_type=db_type
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
    
    def _execute_normalized_sql(self, sql: str):
        """Execute SQL with database-specific normalization and timing."""
        # Normalize SQL outside of timing (this is preprocessing)
        normalized_sql = DatabaseAdapter.normalize_sql_for_db(sql, self.db_type)
        
        # Track only the actual SQL execution time
        try:
            return self._track_sql_execution(lambda: self._execute_sql_direct(normalized_sql))
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common compatibility issues
            if 'INT)' in error_msg or 'AS INT' in error_msg:
                logger.error(f"CAST type error - {self.db_type.upper()} may not support this CAST type. Use SIGNED/UNSIGNED (MySQL) or INTEGER (SQLite/DuckDB)")
            elif 'SUBSTRING_INDEX' in error_msg:
                logger.error(f"SUBSTRING_INDEX not supported in {self.db_type.upper()}")
            elif 'DATEDIFF' in error_msg:
                logger.error(f"DATEDIFF syntax error - {self.db_type.upper()} may require different arguments")
            
            raise
    
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
        self.db_adapter.register(name, df)
    
    def _execute_sql(self, sql: str):
        """Execute SQL with automatic normalization (convenience wrapper)."""
        normalized_sql = DatabaseAdapter.normalize_sql_for_db(sql, self.db_type)
        return self._execute_sql_direct(normalized_sql)
    
    def _execute_sql_direct(self, sql: str):
        """Execute SQL directly without normalization (raw execution)."""
        cursor = self.db_adapter.execute(sql)
        return self.db_adapter.fetch_df(cursor)
    
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
        if use_local:
            if lotus_config['embedding_model'].startswith("litellm_proxy"):
                self.rm = LiteLLMRM(model=lotus_config['embedding_model'])
            else:
                if isinstance(self.rm, SentenceTransformersRM):
                    del self.rm
                self.rm = SentenceTransformersRM(model=lotus_config['embedding_model'])
        else:
            # self.rm = SentenceTransformersRM(model=LOTUS_DEFAULT_RM_MODEL)
            self.rm = LiteLLMRM(model=lotus_config['embedding_model'])

        # self.rm = SentenceTransformersRM(model=LOTUS_DEFAULT_RM_MODEL)
        self.vs = FaissVS()
        lotus.settings.configure(lm=self.lm, rm=self.rm, vs=self.vs)
        
        # Update backends with new configuration
        for backend_name, backend in self.backends.items():
            config = self.llm_config.get_model_config(backend_name)
            backend.set_model_config(
                model=config['model'],
                api_base=config['api_base'],
                api_key=config['api_key'],
                embedding_model=config.get('embedding_model'),
                embedding_api_base=config.get('embedding_api_base'),
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
            backend=self.default_backend,
            max_retries=self.query_generator.max_retries,
            fallback_enabled=self.query_generator.fallback_enabled,
            db_type=self.db_type
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
        max_tokens: int = 512,
        llm_verification: bool = True
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
            llm_verification: Enable LLM-powered verification and optimization (default: True)
            
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
                max_tokens=max_tokens,
                llm_verification=llm_verification
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
                max_tokens=max_tokens,
                llm_verification=llm_verification
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
            self.db_adapter.register(view_name, intersect_df)

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
            self.db_adapter.register(view_name, diff_df)

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
            
        except ValueError as ve:
            # SQL parsing errors from AST rewriter
            error_msg = str(ve)
            if "Failed to parse SQL" in error_msg:
                logging.warning(f"SQL parsing error in AST rewriter: {error_msg}")
                
                # Check if the query has any semantic operations
                # If not, fall back to direct SQL execution
                has_semantic_ops = any(pattern in sql.upper() for pattern in [
                    'SEM_WHERE', 'SEM_SELECT', 'SEM_JOIN', 'SEM_GROUP_BY', 
                    'SEM_AGG', 'SEM_DISTINCT', 'SEM_ORDER_BY', 'SEM_INTERSECT', 'SEM_EXCEPT'
                ])
                
                if not has_semantic_ops:
                    logging.warning("No semantic operations detected, falling back to direct SQL execution")
                    return self._track_sql_execution(lambda: self._execute_sql(sql))
                else:
                    logging.error(f"Problematic SQL with semantic operations: {sql[:500]}...")
                    raise RuntimeError(f"Failed to parse SQL query with semantic operations. This may be due to unsupported SQL syntax or dialect-specific features. Error: {error_msg}")
            else:
                raise
        except Exception as e:
            logging.error(f"Error in AST execution: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _execute_with_semantic_ops(self, canonical_sql: str, operations: List) -> pd.DataFrame:
        # Check if the final SELECT contains a JOIN
        final_select_pattern = r'SELECT\s+.*?\s+FROM\s+.*?\s+JOIN\s+'
        has_final_join = bool(re.search(final_select_pattern, canonical_sql, re.IGNORECASE | re.DOTALL))
        
        # Log canonical SQL for debugging
        logger.debug(f"Executing canonical SQL with {len(operations)} semantic operations")
        logger.debug(f"Canonical SQL preview: {canonical_sql[:200]}...")
        
        # Extract base SQL from CTE more carefully to handle parentheses in SQL statements
        # Use a more robust pattern that looks for the CTE boundary
        cte_start = canonical_sql.find('WITH _child AS (')
        if cte_start != -1:
            # Find the matching closing parenthesis for the CTE
            # Start after 'WITH _child AS ('
            paren_start = cte_start + len('WITH _child AS (')
            paren_count = 1
            i = paren_start
            while i < len(canonical_sql) and paren_count > 0:
                if canonical_sql[i] == '(':
                    paren_count += 1
                elif canonical_sql[i] == ')':
                    paren_count -= 1
                i += 1
            
            if paren_count == 0:
                base_sql = canonical_sql[paren_start:i-1].strip()
            else:
                base_sql = None
        else:
            base_sql = None
        
        if base_sql:
            
            # Check if we have set operations (JOIN, INTERSECT, EXCEPT) - if so, skip base SQL execution
            has_set_operation = any(op.op_type in ('join', 'intersect', 'intersect_all', 'except', 'except_all') 
                                   for op in operations)
            
            # Extract the table name from base_sql to get all columns
            # Pattern: FROM table_name or FROM "table_name" or FROM `table_name` or FROM table_name AS alias
            # Note: This won't match subqueries like FROM (SELECT ...) alias
            table_match = re.search(r'FROM\s+(?:"([^"]+)"|`([^`]+)`|([a-zA-Z_][\w]*))', base_sql, re.IGNORECASE)
            base_table_name = (table_match.group(1) if (table_match and table_match.group(1)) 
                             else (table_match.group(2) if (table_match and table_match.group(2))
                             else (table_match.group(3) if table_match else None)))
            
            # Check if base_sql contains a subquery (base_table_name will be None in this case)
            is_subquery_from = base_table_name is None and 'SELECT' in base_sql.upper()
            
            # If the final SELECT has a JOIN and we have semantic operations, we need to perform JOIN first
            if has_final_join and not has_set_operation:
                # Extract the JOIN from the final SELECT
                # Pattern handles: FROM table1 JOIN table2 [AS alias] ON col1 = col2
                # Updated to handle quoted column names like w."Home team" and backticked names
                join_match = re.search(r'FROM\s+(\w+)\s+JOIN\s+(\w+)(?:\s+AS\s+(\w+))?\s+ON\s+((?:\w+\.)?(?:"[^"]+"|`[^`]+`|[\w]+))\s*=\s*((?:\w+\.)?(?:"[^"]+"|`[^`]+`|[\w]+))', 
                                      canonical_sql, re.IGNORECASE)
                if join_match:
                    # Get the left table name from base_sql
                    left_table_match = re.search(r'FROM\s+(\w+)(?:\s+AS\s+(\w+))?', base_sql, re.IGNORECASE)
                    if left_table_match:
                        left_table_name = left_table_match.group(1)
                        left_alias = left_table_match.group(2) or left_table_name  # Use table name if no alias
                        
                        right_table_name = join_match.group(2)
                        right_alias = join_match.group(3) or right_table_name  # Use table name if no alias
                        left_join_col_raw = join_match.group(4).strip()
                        right_join_col_raw = join_match.group(5).strip()
                        
                        if left_table_name in self._dataframes and right_table_name in self._dataframes:
                            # Execute base_sql to get filtered left table (which includes WHERE clauses)
                            # The base_sql already contains the proper filtering from the original query
                            try:
                                # Extract WHERE clause from base_sql if it exists
                                where_match = re.search(r'WHERE\s+(.+)$', base_sql, re.IGNORECASE | re.DOTALL)
                                if where_match:
                                    where_clause = where_match.group(1).strip()
                                    # Build query with all columns but same WHERE clause
                                    full_query = f"SELECT * FROM {left_table_name} WHERE {where_clause}"
                                    left_df = self._track_sql_execution(lambda: self._execute_sql(full_query))
                                else:
                                    # No WHERE clause in base_sql, use full dataframe
                                    left_df = self._dataframes[left_table_name].copy()
                            except Exception as e:
                                logging.warning(f"Failed to execute base_sql for filtering: {e}, using full dataframe")
                                left_df = self._dataframes[left_table_name].copy()
                            
                            right_df = self._dataframes[right_table_name].copy()
                            
                            # Add prefixes to column names
                            left_df.columns = [f"{left_alias}.{col}" for col in left_df.columns]
                            right_df.columns = [f"{right_alias}.{col}" for col in right_df.columns]
                            
                            # Parse join columns: they may have table/alias prefixes like "documents.title" or "d.name"
                            # We need to determine which column belongs to which dataframe
                            
                            def resolve_column(col_ref, left_tbl, left_al, right_tbl, right_al):
                                """Resolve a column reference to actual column name with correct prefix"""
                                logging.debug(f"resolve_column input: '{col_ref}'")
                                
                                # Handle quoted column names (e.g., w."Home team")
                                # Match: prefix . "quoted col" OR prefix . unquoted_col
                                import re
                                
                                # Try to match: prefix."quoted column"
                                quoted_match = re.match(r'([^.]+)\."([^"]+)"', col_ref)
                                if quoted_match:
                                    prefix = quoted_match.group(1).strip()
                                    col_name = quoted_match.group(2).strip()
                                    logging.debug(f"Matched quoted pattern: prefix='{prefix}', col='{col_name}'")
                                    
                                    # Check if prefix matches left or right table/alias
                                    if prefix in [left_tbl, left_al]:
                                        return f"{left_al}.{col_name}", 'left'
                                    elif prefix in [right_tbl, right_al]:
                                        return f"{right_al}.{col_name}", 'right'
                                    else:
                                        return f"{left_al}.{col_name}", 'left'
                                
                                # Try to match: prefix.unquoted_column
                                unquoted_match = re.match(r'([^.]+)\.(.+)', col_ref)
                                if unquoted_match:
                                    prefix = unquoted_match.group(1).strip()
                                    col_name = unquoted_match.group(2).strip()
                                    logging.debug(f"Matched unquoted pattern: prefix='{prefix}', col='{col_name}'")
                                    
                                    # Check if prefix matches left or right table/alias
                                    if prefix in [left_tbl, left_al]:
                                        return f"{left_al}.{col_name}", 'left'
                                    elif prefix in [right_tbl, right_al]:
                                        return f"{right_al}.{col_name}", 'right'
                                    else:
                                        return f"{left_al}.{col_name}", 'left'
                                
                                # No prefix, just column name (may be quoted)
                                col_ref_clean = col_ref.strip('"').strip("'")
                                logging.debug(f"No prefix, cleaned: '{col_ref_clean}'")
                                return f"{left_al}.{col_ref_clean}", 'left'
                            
                            left_col, left_side = resolve_column(left_join_col_raw, left_table_name, left_alias, right_table_name, right_alias)
                            right_col, right_side = resolve_column(right_join_col_raw, left_table_name, left_alias, right_table_name, right_alias)
                            
                            logging.info(f"Resolved join columns: left='{left_col}' (from {left_side}), right='{right_col}' (from {right_side})")
                            logging.info(f"Left DF columns: {left_df.columns.tolist()}")
                            logging.info(f"Right DF columns: {right_df.columns.tolist()}")
                            
                            # For pandas merge, we need to know which column is from which DataFrame
                            # The JOIN condition might be: ON documents.title = d.name
                            # After resolution: left_col="documents.title" (from right DF), right_col="d.name" (from left DF)
                            # We need to swap them so left_on is from left_df and right_on is from right_df
                            if left_side == 'left' and right_side == 'right':
                                # Standard case: LEFT col = RIGHT col
                                merge_left_on = left_col
                                merge_right_on = right_col
                            elif left_side == 'right' and right_side == 'left':
                                # Swapped: RIGHT col = LEFT col
                                merge_left_on = right_col
                                merge_right_on = left_col
                            elif left_side == 'left' and right_side == 'left':
                                # Both from left - this shouldn't happen in a proper join
                                raise ValueError(f"Both join columns from left table: {left_join_col_raw}, {right_join_col_raw}")
                            else:  # both from right
                                raise ValueError(f"Both join columns from right table: {left_join_col_raw}, {right_join_col_raw}")
                            
                            # Perform the join
                            current_df = left_df.merge(right_df, left_on=merge_left_on, right_on=merge_right_on, how='inner')
                        else:
                            # Fall back to SQL execution
                            def execute_and_fetch():
                                cursor = self.db_adapter.execute(base_sql)
                                return self.db_adapter.fetch_df(cursor)
                            base_result = self._track_sql_execution(execute_and_fetch)
                            current_df = base_result
                    else:
                        # Couldn't parse base_sql, fall back
                        current_df = self._track_sql_execution(lambda: self._execute_sql(base_sql))
                else:
                    # Couldn't parse JOIN, fall back to normal execution
                    current_df = self._track_sql_execution(lambda: self._execute_sql(base_sql))
            elif not has_set_operation:
                # No JOIN in final SELECT, execute base SQL normally
                # But if we have semantic operations, get ALL columns from the base table
                # so that columns referenced in semantic prompts are available
                
                # Check if the final query has GROUP BY or aggregations - if so, we need to be careful about
                # executing base_sql since it might reference aggregated columns that don't exist yet
                has_group_by = bool(re.search(r'GROUP\s+BY\s+', canonical_sql, re.IGNORECASE))
                has_agg_ops = any(op.op_type == 'agg' for op in operations)
                # Also check for traditional SQL aggregations (COUNT, SUM, AVG, etc.) in base_sql or canonical_sql
                has_traditional_agg = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(', canonical_sql, re.IGNORECASE))
                
                # If we have GROUP BY with aggregations and semantic operations, we need special handling
                if is_subquery_from:
                    # For subqueries in FROM clause, execute the base_sql directly
                    # The subquery is self-contained and already has its filtering
                    try:
                        current_df = self._track_sql_execution(lambda: self._execute_sql(base_sql))
                    except Exception as e:
                        logging.error(f"Failed to execute subquery: {e}")
                        raise
                elif operations and base_table_name and base_table_name in self._dataframes and (has_group_by and (has_agg_ops or has_traditional_agg)):
                    # For GROUP BY with aggregations, start with the base table and apply WHERE only
                    current_df = self._dataframes[base_table_name].copy()
                    # Try to apply WHERE clause if present (but not GROUP BY or SELECT with aliases)
                    where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', base_sql, re.IGNORECASE | re.DOTALL)
                    if where_match:
                        where_clause = where_match.group(1).strip()
                        full_query = f"SELECT * FROM {base_table_name} WHERE {where_clause}"
                        try:
                            def execute_and_fetch():
                                cursor = self.db_adapter.execute(full_query)
                                return self.db_adapter.fetch_df(cursor)
                            current_df = self._track_sql_execution(execute_and_fetch)
                        except Exception as e:
                            logging.warning(f"Failed to execute WHERE clause: {e}, using full dataframe")
                elif operations and base_table_name and base_table_name in self._dataframes:
                    # Get all columns from the registered table
                    current_df = self._dataframes[base_table_name].copy()
                    # Apply any WHERE clauses from base_sql if present
                    where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', base_sql, re.IGNORECASE | re.DOTALL)
                    if where_match:
                        # Execute the base SQL to get filtered rows, but select all columns
                        where_clause = where_match.group(1).strip()
                        # Construct a query that selects all columns with the WHERE clause
                        full_query = f"SELECT * FROM {base_table_name} WHERE {where_clause}"
                        try:
                            current_df = self._track_sql_execution(lambda: self._execute_sql(full_query))
                        except Exception as e:
                            # If WHERE clause fails, fall back to full dataframe
                            logging.warning(f"Failed to execute WHERE clause: {e}, using full dataframe")
                            current_df = self._dataframes[base_table_name].copy()
                elif not (has_group_by and (has_agg_ops or has_traditional_agg)):
                    # Only execute base_sql directly if there's no GROUP BY with aggregations
                    # (otherwise base_sql might reference columns that don't exist yet)
                    try:
                        current_df = self._track_sql_execution(lambda: self._execute_sql(base_sql))
                    except Exception as e:
                        # If base_sql fails, try to fall back to the base table
                        if base_table_name and base_table_name in self._dataframes:
                            logging.warning(f"Failed to execute base_sql: {e}, using base table")
                            current_df = self._dataframes[base_table_name].copy()
                        else:
                            raise
                else:
                    # For GROUP BY with aggregations, start with the base table
                    if base_table_name and base_table_name in self._dataframes:
                        current_df = self._dataframes[base_table_name].copy()
                        # Try to apply WHERE clause if present
                        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', base_sql, re.IGNORECASE | re.DOTALL)
                        if where_match:
                            where_clause = where_match.group(1).strip()
                            full_query = f"SELECT * FROM {base_table_name} WHERE {where_clause}"
                            try:
                                current_df = self._track_sql_execution(lambda: self._execute_sql(full_query))
                            except Exception as e:
                                logging.warning(f"Failed to execute WHERE clause: {e}, using full dataframe")
                    else:
                        # Last resort: try to execute base_sql
                        current_df = self._track_sql_execution(lambda: self._execute_sql(base_sql))
            else:
                # For set operations, we'll start with None and the operation will populate it
                current_df = None
            
            # Track if we've done a GROUP BY to inform SEM_AGG
            grouped_by_column = None
            
            # Extract GROUP BY columns from the final SELECT query for SEM_AGG
            group_by_columns = []
            final_query_preview = canonical_sql.split('\n')[-1]
            group_by_match = re.search(r'GROUP\s+BY\s+([^\s]+(?:\s*,\s*[^\s]+)*?)(?:\s+ORDER|\s+HAVING|\s+LIMIT|$)', final_query_preview, re.IGNORECASE)
            if group_by_match:
                # Extract column names from GROUP BY clause
                group_by_str = group_by_match.group(1).strip()
                
                # Handle SEM_SELECT in GROUP BY - look for the alias in the SELECT clause
                if 'SEM_SELECT' in group_by_str.upper():
                    # Find corresponding alias from SELECT clause
                    # Pattern: SEM_SELECT(...) AS alias
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_query_preview, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        select_clause = select_match.group(1)
                        # Look for SEM_SELECT with AS clause
                        sem_select_matches = re.findall(r'SEM_SELECT\s*\([^)]+\)\s*AS\s+(\w+)', select_clause, re.IGNORECASE)
                        if sem_select_matches:
                            group_by_columns = sem_select_matches
                            logger.debug(f"Extracted GROUP BY columns from SEM_SELECT aliases: {group_by_columns}")
                else:
                    # Strip quotes from column names (both " and `)
                    group_by_columns = [col.strip().strip('"').strip('`') for col in group_by_str.split(',') if col.strip()]
                    
                # Filter out empty strings
                group_by_columns = [col for col in group_by_columns if col]
            
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
            self.db_adapter.register('_sem_final', current_df)
            
            # Handle user-defined CTEs - if canonical SQL has user CTEs like:
            # , win_years AS (SELECT yr FROM _sem_2)
            # We need to materialize them by replacing _sem_N with _sem_final
            user_cte_pattern = r',\s+(\w+)\s+AS\s+\(([^)]+)\)\s*(?:SELECT|$)'
            user_cte_matches = list(re.finditer(user_cte_pattern, canonical_sql, re.IGNORECASE))
            for cte_match in user_cte_matches:
                cte_name = cte_match.group(1)
                cte_query = cte_match.group(2).strip()
                # Skip our generated CTEs (_child, _sem_N)
                if cte_name.startswith('_'):
                    continue
                # Replace _sem_N with _sem_final in the CTE query
                cte_query = re.sub(r'_sem_\d+', '_sem_final', cte_query)
                # Execute the CTE query to create the named table
                logger.debug(f"Materializing user CTE '{cte_name}': {cte_query}")
                cte_result = self._execute_sql(cte_query)
                self.db_adapter.register(cte_name, cte_result)
            
            # Log available columns for debugging
            logger.debug(f"Columns in current_df before final SELECT: {current_df.columns.tolist()}")
            
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
                
                # MySQL ONLY_FULL_GROUP_BY compatibility: wrap non-grouped columns in ANY_VALUE()
                if re.search(r'\bGROUP\s+BY\b', final_query, re.IGNORECASE):
                    if self.db_type == 'mysql':
                        # Extract GROUP BY columns
                        group_by_match = re.search(r'GROUP\s+BY\s+([^\s]+)(?:\s+ORDER|\s+LIMIT|$)', final_query, re.IGNORECASE)
                        if group_by_match:
                            group_by_cols = {col.strip() for col in group_by_match.group(1).split(',')}
                            
                            # Extract SELECT columns (before FROM)
                            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_query, re.IGNORECASE | re.DOTALL)
                            if select_match:
                                select_clause = select_match.group(1)
                                
                                # Parse SELECT columns and wrap non-GROUP BY columns in ANY_VALUE()
                                select_parts = []
                                for part in select_clause.split(','):
                                    part = part.strip()
                                    
                                    # Extract alias if present (e.g., "col AS alias" or "func() AS alias")
                                    alias_match = re.search(r'\s+AS\s+(\w+)\s*$', part, re.IGNORECASE)
                                    if alias_match:
                                        alias = alias_match.group(1)
                                        expr = part[:alias_match.start()].strip()
                                    else:
                                        # No alias, the part itself is the column
                                        alias = part
                                        expr = part
                                    
                                    # Check if this is in GROUP BY or is an aggregate function
                                    is_grouped = alias in group_by_cols or expr in group_by_cols
                                    is_aggregate = bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|ANY_VALUE)\s*\(', expr, re.IGNORECASE))
                                    
                                    # Wrap in ANY_VALUE if not grouped and not already aggregated
                                    if not is_grouped and not is_aggregate:
                                        if alias_match:
                                            select_parts.append(f'ANY_VALUE({expr}) AS {alias}')
                                        else:
                                            select_parts.append(f'ANY_VALUE({part})')
                                    else:
                                        select_parts.append(part)
                                
                                # Rebuild SELECT clause
                                new_select = 'SELECT ' + ', '.join(select_parts) + ' FROM'
                                final_query = re.sub(r'SELECT\s+.*?\s+FROM', new_select, final_query, count=1, flags=re.IGNORECASE | re.DOTALL)
                        
                        # Fix ORDER BY to reference ANY_VALUE wrapped columns
                        order_by_match = re.search(r'ORDER\s+BY\s+([^\s,]+)', final_query, re.IGNORECASE)
                        if order_by_match and group_by_match:
                            order_col = order_by_match.group(1).strip()
                            group_by_cols = {col.strip() for col in group_by_match.group(1).split(',')}
                            
                            # If ORDER BY column is not in GROUP BY, wrap it in ANY_VALUE
                            if order_col not in group_by_cols and not re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|ANY_VALUE)\s*\(', order_col, re.IGNORECASE):
                                final_query = re.sub(
                                    r'ORDER\s+BY\s+' + re.escape(order_col),
                                    f'ORDER BY ANY_VALUE({order_col})',
                                    final_query,
                                    flags=re.IGNORECASE
                                )
                    else:
                        # For non-MySQL databases, just wrap SEM_AGG aliases
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
                        # Updated pattern to handle quoted column names
                        final_query = re.sub(
                            r'FROM\s+_sem_final\s+JOIN\s+\w+(?:\s+AS\s+\w+)?\s+ON\s+((?:\w+\.)?"[^"]+"|(?:\w+\.)?[\w]+)\s*=\s*((?:\w+\.)?"[^"]+"|(?:\w+\.)?[\w]+)',
                            'FROM _sem_final',
                            final_query,
                            flags=re.IGNORECASE
                        )
                
                # Quote columns with dots for proper SQL execution
                final_query = quote_dot_columns(final_query, current_df.columns.tolist(), self.db_type)
                
                # Remove any remaining semantic function calls from the final query
                # These should have been processed already, but sometimes they appear in GROUP BY or ORDER BY
                sem_func_pattern = r'\bSEM_(SELECT|WHERE|AGG|ORDER_BY|GROUP_BY|JOIN|DISTINCT|INTERSECT|EXCEPT)\s*\([^)]*\)'
                if re.search(sem_func_pattern, final_query, re.IGNORECASE):
                    logger.warning(f"Found semantic function calls in final SQL query - these should have been removed by AST rewriter")
                    logger.warning(f"Query with semantic functions: {final_query}")
                    
                    # For GROUP BY, use the already-extracted group_by_columns if available
                    # This prevents empty GROUP BY clauses
                    group_by_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+ORDER|\s+HAVING|\s+LIMIT|$)', final_query, re.IGNORECASE)
                    if group_by_match:
                        group_by_clause = group_by_match.group(1).strip()
                        if 'SEM_SELECT' in group_by_clause.upper() or 'SEM_GROUP_BY' in group_by_clause.upper():
                            # Use the columns extracted earlier in the execution loop
                            if group_by_columns:
                                new_group_by = ', '.join(group_by_columns)
                                final_query = re.sub(
                                    r'GROUP\s+BY\s+.*?(?=\s+ORDER|\s+HAVING|\s+LIMIT|$)',
                                    f'GROUP BY {new_group_by}',
                                    final_query,
                                    count=1,
                                    flags=re.IGNORECASE
                                )
                                logger.info(f"Replaced GROUP BY with extracted columns: {new_group_by}")
                            else:
                                # Fallback: try to extract alias from SELECT clause
                                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_query, re.IGNORECASE | re.DOTALL)
                                if select_match:
                                    select_clause = select_match.group(1)
                                    # Find SEM_SELECT or SEM_GROUP_BY with AS alias
                                    sem_alias_matches = re.findall(r'(?:SEM_SELECT|SEM_GROUP_BY)\s*\([^)]+\)\s*AS\s+(\w+)', select_clause, re.IGNORECASE)
                                    if sem_alias_matches:
                                        new_group_by = ', '.join(sem_alias_matches)
                                        final_query = re.sub(
                                            r'GROUP\s+BY\s+.*?(?=\s+ORDER|\s+HAVING|\s+LIMIT|$)',
                                            f'GROUP BY {new_group_by}',
                                            final_query,
                                            count=1,
                                            flags=re.IGNORECASE
                                        )
                                        logger.info(f"Replaced GROUP BY with aliases from SELECT: {new_group_by}")
                                    else:
                                        logger.error("Cannot find GROUP BY columns - removing GROUP BY clause")
                                        final_query = re.sub(r'GROUP\s+BY\s+.*?(?=\s+ORDER|\s+HAVING|\s+LIMIT|$)', '', final_query, count=1, flags=re.IGNORECASE)
                    
                    # For ORDER BY, similar replacement
                    final_query = re.sub(
                        r'ORDER\s+BY\s+SEM_SELECT\s*\([^)]+\)\s*AS\s+(\w+)',
                        r'ORDER BY \1',
                        final_query,
                        flags=re.IGNORECASE
                    )
                    
                    # Additional GROUP BY fix: replace any remaining SEM_* functions in GROUP BY with actual column names from _sem_final
                    group_by_match = re.search(r'GROUP\s+BY\s+(.*?)(?=\s+ORDER|\s+HAVING|\s+LIMIT|$)', final_query, re.IGNORECASE | re.DOTALL)
                    if group_by_match and current_df is not None:
                        group_by_clause = group_by_match.group(1)
                        # Check if GROUP BY still contains SEM_* functions
                        if re.search(r'SEM_\w+\s*\(', group_by_clause, re.IGNORECASE):
                            logger.warning(f"GROUP BY still contains semantic functions after initial replacement: {group_by_clause}")
                            # Try to match semantic functions in GROUP BY with columns in current_df
                            # Extract all SEM_* function calls in GROUP BY
                            sem_funcs_in_group = re.findall(r'SEM_\w+\s*\([^)]+\)', group_by_clause, re.IGNORECASE)
                            if sem_funcs_in_group:
                                # For each SEM function, try to find the corresponding column in SELECT clause
                                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_query, re.IGNORECASE | re.DOTALL)
                                if select_match:
                                    select_clause = select_match.group(1)
                                    replacement_map = {}
                                    for sem_func in sem_funcs_in_group:
                                        # Find this same function in SELECT with an alias
                                        alias_match = re.search(rf'{re.escape(sem_func)}\s+AS\s+(\w+)', select_clause, re.IGNORECASE)
                                        if alias_match:
                                            replacement_map[sem_func] = alias_match.group(1)
                                            logger.info(f"Mapping GROUP BY function {sem_func} to column {alias_match.group(1)}")
                                    
                                    # Apply replacements
                                    if replacement_map:
                                        for sem_func, col_name in replacement_map.items():
                                            group_by_clause = group_by_clause.replace(sem_func, col_name)
                                        final_query = re.sub(
                                            r'GROUP\s+BY\s+.*?(?=\s+ORDER|\s+HAVING|\s+LIMIT|$)',
                                            f'GROUP BY {group_by_clause}',
                                            final_query,
                                            count=1,
                                            flags=re.IGNORECASE
                                        )
                                        logger.info(f"Replaced GROUP BY semantic functions with column names: {group_by_clause}")
                    
                    # Remove standalone SEM_WHERE from SELECT clause (common error)
                    # SEM_WHERE should only be in WHERE clause, not SELECT
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_query, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        select_clause = select_match.group(1)
                        if 'SEM_WHERE' in select_clause.upper():
                            logger.error("SEM_WHERE found in SELECT clause - this is invalid. Removing it.")
                            # Remove SEM_WHERE from SELECT, keeping other columns
                            cleaned_select = re.sub(r',?\s*SEM_WHERE\s*\([^)]*\)\s*(?:AS\s+\w+)?\s*,?', '', select_clause, flags=re.IGNORECASE)
                            cleaned_select = re.sub(r'^\s*,|,\s*$', '', cleaned_select).strip()  # Clean up extra commas
                            if not cleaned_select or cleaned_select == ',':
                                cleaned_select = '*'  # Fallback to SELECT *
                            final_query = re.sub(r'SELECT\s+.*?\s+FROM', f'SELECT {cleaned_select} FROM', final_query, count=1, flags=re.IGNORECASE | re.DOTALL)
                    
                    # Remove other standalone SEM_* functions that shouldn't be there
                    final_query = re.sub(sem_func_pattern, '', final_query, flags=re.IGNORECASE)
                    
                    # Clean up any resulting double spaces or trailing commas
                    final_query = re.sub(r'\s+', ' ', final_query)
                    final_query = re.sub(r',\s*,', ',', final_query)
                    final_query = re.sub(r',\s*FROM', ' FROM', final_query, flags=re.IGNORECASE)
                    
                    logger.info(f"Cleaned final query: {final_query}")
                
                # Remove MySQL-incompatible syntax
                if self.db_type == 'mysql':
                    # MySQL doesn't support NULLS FIRST/LAST
                    final_query = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', final_query, flags=re.IGNORECASE)
                
                # Execute final query with error handling
                try:
                    result = self._execute_normalized_sql(final_query)
                except Exception as e:
                    error_msg = str(e)
                    
                    # Handle "Unknown column" errors - these might be from semantic operations
                    if 'Unknown column' in error_msg or 'not found' in error_msg.lower():
                        logger.warning(f"Column resolution error: {error_msg}")
                        logger.warning(f"Available columns in _sem_final: {current_df.columns.tolist()}")
                        logger.warning(f"Final query attempted: {final_query}")
                        
                        # Try to extract what columns are actually being selected
                        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', final_query, re.IGNORECASE | re.DOTALL)
                        if select_match:
                            requested_cols = [col.strip() for col in select_match.group(1).split(',')]
                            logger.warning(f"Requested columns: {requested_cols}")
                            
                            # Check if any requested columns don't exist in current_df
                            missing_cols = [col for col in requested_cols if col not in current_df.columns and col != '*' and col != 'DISTINCT']
                            if missing_cols:
                                logger.error(f"Missing columns in result DataFrame: {missing_cols}")
                                logger.error(f"This likely means the backend semantic operation failed to create the expected columns")
                                # Return the current_df directly instead of trying to SELECT
                                logger.warning("Returning current DataFrame directly instead of executing SELECT")
                                return current_df
                            
                            # Fallback: just select all columns from current_df
                            logger.warning("Falling back to SELECT * FROM _sem_final")
                            fallback_query = re.sub(r'SELECT\s+.*?\s+FROM', 'SELECT * FROM', final_query, count=1, flags=re.IGNORECASE | re.DOTALL)
                            result = self._execute_normalized_sql(fallback_query)
                        else:
                            raise
                    elif 'does not exist' in error_msg and 'FUNCTION' in error_msg:
                        # Function doesn't exist (e.g., unsupported database-specific function)
                        logger.error(f"Database function not supported in {self.db_type}: {error_msg}")
                        logger.error(f"This likely means the query used a function that couldn't be normalized. "
                                   f"Consider using semantic operations (SEM_SELECT, SEM_WHERE, etc.) instead.")
                        raise
                    elif 'syntax' in error_msg.lower() and ('int)' in error_msg.lower() or 'cast' in error_msg.lower()):
                        # CAST type compatibility issue
                        logger.error(f"CAST type incompatibility with {self.db_type}: {error_msg}")
                        logger.error(f"The query may be using incorrect type casting syntax for {self.db_type}")
                        raise
                    else:
                        raise
                return result
            else:
                return current_df
        else:
            return self._track_sql_execution(lambda: self._execute_sql(canonical_sql))
    
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
            
            # 2. Semantic WHERE clauses (apply each filter sequentially)
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
            self.db_adapter.register(temp_table, df)

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
                for _ in semantic_ops['where']:
                    modified_sql = re.sub(self.sql_parser.patterns['where'], "TRUE", modified_sql, count=1)
            
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
            modified_sql = quote_dot_columns(modified_sql, df, self.db_type)
            logging.info(f"Final modified SQL for execution:\n{modified_sql}\n")

            result = self._track_sql_execution(lambda: self.db_adapter.sql(modified_sql).df())

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
                if self.db_type == 'duckdb':
                    self.db_adapter.execute(f"DROP VIEW IF EXISTS {temp_table}")
                else:
                    self.db_adapter.execute(f"DROP TABLE IF EXISTS {temp_table}")
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
                if self.db_type == 'duckdb':
                    self.db_adapter.execute(f"DROP VIEW IF EXISTS {temp_table_name}")
                else:
                    self.db_adapter.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
                if temp_table_name in self._dataframes:
                    del self._dataframes[temp_table_name]
            except:
                pass
        
        # Execute final SQL
        return self._track_sql_execution(lambda: self.db_adapter.sql(sql).df())
    
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

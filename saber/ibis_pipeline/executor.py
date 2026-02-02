"""
Pipeline Executor for Ibis + DuckDB + pandas.

Orchestrates execution of segmented query plans:
1. Execute relational segments in DuckDB
2. Bridge results to pandas
3. Execute semantic segments with SDPS backends
4. Bridge back to DuckDB for next relational segment
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List
import pandas as pd

from .logical_plan import LogicalPlan, PlanNode, RelationalNode, SemanticNode, ScanNode
from .segmenter import PlanSegmenter, Segment, RelationalSegment, SemanticSegment
from .optimizer import PlanOptimizer
from .parser import IbisSQLParser
from .bridge import ToPandas, FromPandas
from .operators import SemJoin, SemIntersect, SemExcept, SemFilterMarker

logger = logging.getLogger(__name__)


class ExecutionStats:
    """Statistics from pipeline execution."""
    
    def __init__(self):
        self.total_time_ms: float = 0.0
        self.duckdb_time_ms: float = 0.0
        self.pandas_time_ms: float = 0.0
        self.bridge_time_ms: float = 0.0
        self.semantic_time_ms: float = 0.0
        self.segments_executed: int = 0
        self.bridge_crossings: int = 0


class PipelineExecutor:
    """Executes hybrid relational-semantic query plans."""
    
    def __init__(self, db_adapter: Any, backends: Dict[str, Any],
                 dataframes: Dict[str, pd.DataFrame],
                 semantic_ops: Any = None, rm: Any = None):
        self.db_adapter = db_adapter
        self.backends = backends
        self.dataframes = dataframes
        self.semantic_ops = semantic_ops
        self.rm = rm
        
        self.parser = IbisSQLParser()
        self.optimizer = PlanOptimizer()
        self.segmenter = PlanSegmenter()
        
        self.to_pandas = ToPandas()
        self.from_pandas = FromPandas()
        
        self.stats = ExecutionStats()
        self._temp_view_counter = 0
        self._segment_results: Dict[int, pd.DataFrame] = {}
    
    def execute(self, sql: str, explain: bool = False) -> pd.DataFrame:
        """Execute a SQL query with semantic operations."""
        start_time = time.time()
        self.stats = ExecutionStats()
        self._segment_results = {}
        self._temp_view_counter = 0
        
        # Register table schemas with parser
        for name, df in self.dataframes.items():
            self.parser.set_table_schema(name, list(df.columns))
        
        # Parse SQL into logical plan
        plan = self.parser.parse(sql)
        
        if explain:
            logger.info("=== Original Logical Plan ===")
            logger.info('\n' + plan.explain())
        
        # Optimize the plan
        optimized_plan = self.optimizer.optimize(plan)
        
        if explain:
            logger.info("=== Optimized Logical Plan ===")
            logger.info('\n' + optimized_plan.explain())
        
        # Check if we have semantic operations
        if not optimized_plan.has_semantic_ops():
            # Pure relational query - execute directly in DuckDB
            result = self._execute_pure_sql(sql)
        else:
            # Segment and execute hybrid plan
            segments = self.segmenter.segment(optimized_plan)
            
            if explain:
                logger.info('\n' + self.segmenter.explain())
            
            # Execute segments in dependency order
            result = self._execute_segments(segments, sql)
        
        self.stats.total_time_ms = (time.time() - start_time) * 1000
        
        if explain:
            logger.info(f"=== Execution Stats ===")
            logger.info(f"Total: {self.stats.total_time_ms:.2f}ms")
            logger.info(f"DuckDB: {self.stats.duckdb_time_ms:.2f}ms")
            logger.info(f"Semantic: {self.stats.semantic_time_ms:.2f}ms")
            logger.info(f"Bridge: {self.stats.bridge_time_ms:.2f}ms")
            logger.info(f"Segments: {self.stats.segments_executed}")
            logger.info(f"Bridge crossings: {self.stats.bridge_crossings}")
        
        return result
    
    def _execute_pure_sql(self, sql: str) -> pd.DataFrame:
        """Execute a pure SQL query without semantic operations."""
        start = time.time()
        cursor = self.db_adapter.execute(sql)
        result = self.db_adapter.fetch_df(cursor)
        self.stats.duckdb_time_ms = (time.time() - start) * 1000
        return result
    
    def _execute_segments(self, segments: List[Segment], original_sql: str) -> pd.DataFrame:
        """Execute segments in topological order."""
        execution_order = self.segmenter.get_execution_order()
        
        for seg_id in execution_order:
            segment = self._get_segment(segments, seg_id)
            if not segment:
                continue
            
            # Check if this is the final segment
            is_final_segment = (seg_id == execution_order[-1])
            
            if isinstance(segment, RelationalSegment):
                result = self._execute_relational_segment(segment, original_sql, is_final_segment)
            elif isinstance(segment, SemanticSegment):
                result = self._execute_semantic_segment(segment)
                # If semantic segment is the final segment, apply final projections/aggregations
                if is_final_segment:
                    result = self._apply_final_projections(result, original_sql)
            else:
                continue
            
            self._segment_results[seg_id] = result
            self.stats.segments_executed += 1
        
        # Return the result from the final segment
        if execution_order:
            return self._segment_results.get(execution_order[-1], pd.DataFrame())
        return pd.DataFrame()
    
    def _apply_final_projections(self, df: pd.DataFrame, original_sql: str) -> pd.DataFrame:
        """Apply final SELECT projections/aggregations when semantic segment is final.
        
        This handles cases like COUNT(*), SUM(), etc. that need to be applied
        after the semantic operation completes, as well as simple column projections.
        """
        import re
        from sqlglot import parse_one
        
        if df.empty:
            return df
        
        try:
            ast = parse_one(original_sql, dialect='duckdb')
            
            # Check if there are aggregate functions in SELECT
            has_aggregates = False
            if hasattr(ast, 'expressions') and ast.expressions:
                for expr in ast.expressions:
                    col_sql = expr.sql(dialect='duckdb')
                    # Check for common aggregate functions
                    if re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(', col_sql, re.IGNORECASE):
                        has_aggregates = True
                        break
            
            # Register the DataFrame and execute in DuckDB for both aggregates and simple projections
            temp_table = f"_ibis_final_agg_{self._temp_view_counter}"
            self._temp_view_counter += 1
            self.from_pandas.execute(df, self.db_adapter, temp_table)
            
            # Build the final SQL with projections
            select_cols = []
            if hasattr(ast, 'expressions') and ast.expressions:
                for expr in ast.expressions:
                    col_sql = expr.sql(dialect='duckdb')
                    # Replace SEM_WHERE(...) with _sem_where_result column
                    # This handles CASE WHEN SEM_WHERE(...) THEN 1 ELSE 0 END patterns
                    # The _sem_where_result column was added by SemFilterMarker
                    if 'SEM_WHERE' in col_sql.upper():
                        col_sql = re.sub(
                            r"SEM_WHERE\s*\([^)]*(?:\([^)]*\)[^)]*)*\)",
                            "_sem_where_result",
                            col_sql,
                            flags=re.IGNORECASE
                        )
                    # Skip standalone SEM_SELECT - get alias only
                    elif 'SEM_SELECT' in col_sql.upper():
                        alias_match = re.search(r'\s+AS\s+(\w+)', col_sql, re.IGNORECASE)
                        if alias_match:
                            col_sql = alias_match.group(1)
                        else:
                            continue
                    select_cols.append(col_sql)
            
            if not select_cols:
                return df
            
            # Build and execute the projection/aggregation query
            final_sql = f"SELECT {', '.join(select_cols)} FROM {temp_table}"
            
            # Add GROUP BY if present in original
            if hasattr(ast, 'args') and 'group' in ast.args and ast.args['group']:
                group_exprs = [e.sql(dialect='duckdb') for e in ast.args['group'].expressions]
                final_sql += f" GROUP BY {', '.join(group_exprs)}"
            
            start = time.time()
            result = self._execute_sql(final_sql)
            self.stats.duckdb_time_ms += (time.time() - start) * 1000
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to apply final projections: {e}")
            return df
    
    def _get_segment(self, segments: List[Segment], seg_id: int) -> Optional[Segment]:
        """Get segment by ID."""
        for seg in segments:
            if seg.segment_id == seg_id:
                return seg
        return None
    
    def _execute_relational_segment(self, segment: RelationalSegment,
                                   original_sql: str,
                                   is_final_segment: bool = False) -> pd.DataFrame:
        """Execute a relational segment in DuckDB."""
        start = time.time()
        
        # Check if this segment depends on a previous segment (e.g., final projections after semantic ops)
        if segment.depends_on:
            # Get the result from the dependency
            dep_result = self._segment_results.get(segment.depends_on[0])
            if dep_result is not None and not dep_result.empty:
                # Register the dependency result as a temporary table (bridge operation)
                bridge_start = time.time()
                temp_table = f"_ibis_temp_{segment.segment_id}"
                self.from_pandas.execute(dep_result, self.db_adapter, temp_table)
                self.stats.bridge_time_ms += (time.time() - bridge_start) * 1000
                
                # If there are relational operations to apply, build and execute SQL
                duckdb_start = time.time()
                if segment.nodes:
                    sql = self._build_segment_sql(segment, original_sql, source_table=temp_table,
                                                  is_final_segment=is_final_segment)
                    result = self._execute_sql(sql)
                else:
                    # No operations, just pass through
                    result = dep_result
                
                self.stats.duckdb_time_ms += (time.time() - duckdb_start) * 1000
                return result
        
        # For scan nodes, just return the registered table
        if len(segment.nodes) == 1 and isinstance(segment.nodes[0], ScanNode):
            scan_node = segment.nodes[0]
            table_name = scan_node.table_name
            
            # Check for subquery placeholders
            if table_name.startswith('__subquery_'):
                # Extract the actual subquery and execute it
                if '_left_' in table_name:
                    subquery = table_name.replace('__subquery_left_', '')
                else:
                    subquery = table_name.replace('__subquery_right_', '')
                
                result = self._evaluate_subquery(subquery)
            elif table_name in self.dataframes:
                result = self.dataframes[table_name].copy()
            else:
                # Try to execute as SQL
                result = self._execute_sql(f"SELECT * FROM {table_name}")
        else:
            # Build SQL for relational operations
            sql = self._build_segment_sql(segment, original_sql, is_final_segment=is_final_segment)
            result = self._execute_sql(sql)
        
        self.stats.duckdb_time_ms += (time.time() - start) * 1000
        return result
    
    def _evaluate_subquery(self, query: str) -> pd.DataFrame:
        """Evaluate a subquery (either SQL or table reference)."""
        query = query.strip()
        
        # Remove outer parens if present
        if query.startswith('(') and query.endswith(')'):
            query = query[1:-1].strip()
        
        # Remove outer quotes if present
        if query.startswith("'") and query.endswith("'"):
            query = query[1:-1]
        elif query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        
        # Check if it's a table name
        if query in self.dataframes:
            return self.dataframes[query].copy()
        
        # Check if it's a SELECT query
        if query.upper().startswith('SELECT'):
            return self.execute(query)
        
        raise ValueError(f"Cannot evaluate subquery: {query}")
    
    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL in DuckDB."""
        try:
            cursor = self.db_adapter.execute(sql)
            return self.db_adapter.fetch_df(cursor)
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            logger.error(f"SQL: {sql[:500]}")
            raise
    
    def _quote_prefixed_column(self, col_expr: str) -> str:
        """Quote column expressions that have table prefixes (e.g., r.name -> "r.name").
        
        When a JOIN creates columns with prefixed names, we need to quote them
        for DuckDB to recognize them as literal column names instead of table.column.
        """
        import re
        # Pattern to match table.column references (e.g., r.name, c.rating)
        # But NOT function calls like CAST(r.rating AS FLOAT)
        prefix_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        def replace_prefix(match):
            full_match = match.group(0)
            table_alias = match.group(1)
            col_name = match.group(2)
            # Don't quote if it looks like a function or keyword
            if table_alias.upper() in ('AS', 'AND', 'OR', 'NOT', 'CAST', 'FLOAT', 'INT', 'REAL', 'TEXT'):
                return full_match
            return f'"{table_alias}.{col_name}"'
        
        return re.sub(prefix_pattern, replace_prefix, col_expr)
    
    def _build_segment_sql(self, segment: RelationalSegment, original_sql: str, 
                          source_table: Optional[str] = None,
                          is_final_segment: bool = False) -> str:
        """Build SQL for a relational segment.
        
        Args:
            segment: The relational segment to build SQL for
            original_sql: The original SQL query
            source_table: The temp table name if this segment depends on a previous segment
            is_final_segment: If True, apply SELECT projection, ORDER BY, and LIMIT from original SQL
                              If False, just pass through all columns (SELECT *)
        """
        import re
        from sqlglot import parse_one
        
        # Check if we're dealing with a joined table (columns have prefix like r.name)
        has_prefixed_columns = source_table and source_table.startswith('_ibis_temp_')
        
        # If a source table is provided (from previous segment), use it
        if source_table:
            # For intermediate segments, just pass through all columns
            if not is_final_segment:
                return f"SELECT * FROM {source_table}"
            
            # Extract predicates from filter nodes in this segment
            where_clauses = []
            limit_val = None
            for node in segment.nodes:
                if isinstance(node, RelationalNode):
                    if node.op_type == 'filter' and node.predicate:
                        # Clean up predicate - remove TRUE placeholders from SEM_WHERE replacement
                        pred = node.predicate
                        # Remove standalone TRUE with AND/OR
                        pred = re.sub(r'\s+AND\s+TRUE\b', '', pred, flags=re.IGNORECASE)
                        pred = re.sub(r'\bTRUE\s+AND\s+', '', pred, flags=re.IGNORECASE)
                        pred = re.sub(r'\s+OR\s+TRUE\b', '', pred, flags=re.IGNORECASE)
                        pred = re.sub(r'\bTRUE\s+OR\s+', '', pred, flags=re.IGNORECASE)
                        # Remove standalone TRUE
                        pred = pred.strip()
                        if pred.upper() != 'TRUE' and pred:
                            # Quote prefixed columns if from joined table
                            if has_prefixed_columns:
                                pred = self._quote_prefixed_column(pred)
                            where_clauses.append(pred)
                    elif node.op_type == 'limit' and node.limit:
                        limit_val = node.limit
            
            # Parse the original SQL to extract SELECT columns and ORDER BY
            try:
                ast = parse_one(original_sql, dialect='duckdb')
                
                # Extract SELECT columns
                select_cols = []
                if hasattr(ast, 'expressions') and ast.expressions:
                    for expr in ast.expressions:
                        col_sql = expr.sql(dialect='duckdb')
                        # Skip SEM_SELECT - get alias only
                        if 'SEM_SELECT' in col_sql.upper():
                            # Extract alias after AS
                            alias_match = re.search(r'\s+AS\s+(\w+)', col_sql, re.IGNORECASE)
                            if alias_match:
                                select_cols.append(alias_match.group(1))
                        else:
                            # Quote prefixed columns if from joined table
                            if has_prefixed_columns:
                                col_sql = self._quote_prefixed_column(col_sql)
                            select_cols.append(col_sql)
                
                # Build SELECT clause
                if select_cols:
                    select_clause = ', '.join(select_cols)
                else:
                    select_clause = '*'
                
                # Build WHERE clause from segment nodes
                where_clause = ""
                if where_clauses:
                    where_clause = f" WHERE {' AND '.join(where_clauses)}"
                
                # Extract ORDER BY if present
                order_by = ""
                if hasattr(ast, 'args') and 'order' in ast.args and ast.args['order']:
                    order_exprs = ast.args['order'].expressions
                    order_cols = []
                    for expr in order_exprs:
                        col_sql = expr.sql(dialect='duckdb')
                        # Skip SEM_ORDER_BY
                        if 'SEM_ORDER_BY' not in col_sql.upper():
                            # Quote prefixed columns if from joined table
                            if has_prefixed_columns:
                                col_sql = self._quote_prefixed_column(col_sql)
                            order_cols.append(col_sql)
                    if order_cols:
                        order_by = f" ORDER BY {', '.join(order_cols)}"
                
                # Use limit from segment nodes if available, otherwise from original SQL
                limit_clause = ""
                if limit_val:
                    limit_clause = f" LIMIT {limit_val}"
                elif hasattr(ast, 'args') and 'limit' in ast.args and ast.args['limit']:
                    limit_sql = ast.args['limit'].sql(dialect='duckdb')
                    # The limit value might include 'LIMIT' keyword already, check and handle
                    if limit_sql.upper().startswith('LIMIT'):
                        limit_clause = f" {limit_sql}"
                    else:
                        limit_clause = f" LIMIT {limit_sql}"
                
                return f"SELECT {select_clause} FROM {source_table}{where_clause}{order_by}{limit_clause}"
                
            except Exception as e:
                logger.warning(f"Failed to parse SQL for segment building: {e}")
                return f"SELECT * FROM {source_table}"
        
        # Original table-based logic (first segment with no source_table)
        if segment.tables_referenced:
            table = segment.tables_referenced[0]
            
            # Extract predicates from filter nodes in this segment
            where_clauses = []
            for node in segment.nodes:
                if isinstance(node, RelationalNode) and node.op_type == 'filter' and node.predicate:
                    # Clean up predicate - remove TRUE placeholders from SEM_WHERE replacement
                    pred = node.predicate
                    pred = re.sub(r'\s+AND\s+TRUE\b', '', pred, flags=re.IGNORECASE)
                    pred = re.sub(r'\bTRUE\s+AND\s+', '', pred, flags=re.IGNORECASE)
                    pred = re.sub(r'\s+OR\s+TRUE\b', '', pred, flags=re.IGNORECASE)
                    pred = re.sub(r'\bTRUE\s+OR\s+', '', pred, flags=re.IGNORECASE)
                    pred = pred.strip()
                    if pred.upper() != 'TRUE' and pred:
                        where_clauses.append(pred)
            
            if where_clauses:
                return f"SELECT * FROM {table} WHERE {' AND '.join(where_clauses)}"
            return f"SELECT * FROM {table}"
        
        return "SELECT 1"
    
    def _execute_semantic_segment(self, segment: SemanticSegment) -> pd.DataFrame:
        """Execute a semantic segment with SDPS backend."""
        segment_start = time.time()
        
        # Get input from dependent segment
        if segment.depends_on:
            current_df = self._segment_results.get(segment.depends_on[0], pd.DataFrame())
            # Count as 1 bridge crossing for left/main input
            self.stats.bridge_crossings += 1
        else:
            current_df = pd.DataFrame()
        
        # Get right input for binary ops
        right_df = None
        if segment.right_input_segment:
            right_df = self._segment_results.get(segment.right_input_segment, pd.DataFrame())
            # Count as 1 bridge crossing for right input (binary ops)
            self.stats.bridge_crossings += 1
        
        # Execute semantic operations
        backend = self.backends.get(segment.backend, self.backends.get('lotus'))
        
        for op in segment.operations:
            op_start = time.time()
            
            if isinstance(op, SemJoin):
                # Binary operation - need both inputs
                if not current_df.empty and right_df is not None and not right_df.empty:
                    current_df = op.execute(current_df, right_df, backend)
                elif right_df is not None and not right_df.empty:
                    # Get left from dataframes
                    left_name = op.left_table
                    right_name = op.right_table
                    if left_name in self.dataframes and right_name in self.dataframes:
                        current_df = op.execute(
                            self.dataframes[left_name],
                            self.dataframes[right_name],
                            backend
                        )
            elif isinstance(op, (SemIntersect, SemExcept)):
                # Set operations
                if not current_df.empty and right_df is not None:
                    current_df = op.execute(current_df, right_df, self.rm, self.semantic_ops)
            else:
                # Unary operations
                if current_df.empty:
                    # Try to get data from referenced tables
                    # This happens when scan nodes reference tables
                    continue
                current_df = op.execute(current_df, backend)
            
            self.stats.semantic_time_ms += (time.time() - op_start) * 1000
        
        # Register result for potential future DuckDB access
        bridge_start = time.time()
        view_name = self._next_temp_view()
        self.from_pandas.execute(current_df, self.db_adapter, view_name)
        self.stats.bridge_time_ms += (time.time() - bridge_start) * 1000
        self.stats.bridge_crossings += 1  # Count as 1 bridge crossing for output
        
        return current_df
    
    def _next_temp_view(self) -> str:
        """Generate next temporary view name."""
        self._temp_view_counter += 1
        return f"_ibis_temp_{self._temp_view_counter}"


class DirectSemanticExecutor:
    """Simplified executor that directly handles semantic operations.
    
    This is used for cases where the full pipeline is overkill and
    we just need to execute semantic operations on registered tables.
    """
    
    def __init__(self, db_adapter: Any, backends: Dict[str, Any],
                 dataframes: Dict[str, pd.DataFrame],
                 semantic_ops: Any = None, rm: Any = None):
        self.db_adapter = db_adapter
        self.backends = backends
        self.dataframes = dataframes
        self.semantic_ops = semantic_ops
        self.rm = rm
        
        self.to_pandas = ToPandas()
        self.from_pandas = FromPandas()
        self._temp_counter = 0
    
    def execute_sem_where(self, table: str, prompt: str, backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_WHERE on a table."""
        df = self.dataframes.get(table)
        if df is None:
            raise ValueError(f"Table '{table}' not found")
        
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_where(df, prompt)
    
    def execute_sem_select(self, table: str, prompt: str, alias: str,
                          backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_SELECT on a table."""
        df = self.dataframes.get(table)
        if df is None:
            raise ValueError(f"Table '{table}' not found")
        
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_select(df, prompt, alias)
    
    def execute_sem_join(self, left_table: str, right_table: str,
                        prompt: str, backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_JOIN on two tables."""
        df1 = self.dataframes.get(left_table)
        df2 = self.dataframes.get(right_table)
        
        if df1 is None:
            raise ValueError(f"Table '{left_table}' not found")
        if df2 is None:
            raise ValueError(f"Table '{right_table}' not found")
        
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_join(df1, df2, prompt, left_table, right_table)
    
    def execute_sem_group_by(self, table: str, column: str, k: int = 8,
                            backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_GROUP_BY on a table."""
        df = self.dataframes.get(table)
        if df is None:
            raise ValueError(f"Table '{table}' not found")
        
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_group_by(df, column, k)
    
    def execute_sem_agg(self, df: pd.DataFrame, prompt: str, alias: str,
                       group_by: Optional[List[str]] = None, column: Optional[str] = None,
                       backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_AGG on a DataFrame."""
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_agg(df, prompt, alias, group_by, column)
    
    def execute_sem_distinct(self, table: str, column: str,
                            backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_DISTINCT on a table."""
        df = self.dataframes.get(table)
        if df is None:
            raise ValueError(f"Table '{table}' not found")
        
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_distinct(df, column)
    
    def execute_sem_order_by(self, df: pd.DataFrame, prompt: str,
                            column: Optional[str] = None,
                            backend: str = 'lotus') -> pd.DataFrame:
        """Execute SEM_ORDER_BY on a DataFrame."""
        backend_impl = self.backends.get(backend, self.backends.get('lotus'))
        return backend_impl.sem_order_by(df, prompt, column)
    
    def execute_sem_intersect(self, df1: pd.DataFrame, df2: pd.DataFrame,
                             is_all: bool = False) -> pd.DataFrame:
        """Execute SEM_INTERSECT on two DataFrames."""
        return self.semantic_ops.intersect_operation(df1, df2, self.rm, is_set=not is_all)
    
    def execute_sem_except(self, df1: pd.DataFrame, df2: pd.DataFrame,
                          is_all: bool = False) -> pd.DataFrame:
        """Execute SEM_EXCEPT on two DataFrames."""
        return self.semantic_ops.except_operation(df1, df2, self.rm, is_set=not is_all)

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sqlglot import parse_one, exp
from sqlglot.optimizer import optimize


@dataclass
class SemanticOperation:
    op_type: str
    args: List[str]
    position: int
    
    def __repr__(self):
        return f"SemanticOp({self.op_type}, args={self.args})"


class PendingOperations:
    def __init__(self):
        self.operations: List[SemanticOperation] = []
        self._counter = 0
    
    def add(self, op_type: str, args: List[str]):
        self.operations.append(SemanticOperation(
            op_type=op_type,
            args=args,
            position=self._counter
        ))
        self._counter += 1
    
    def clear(self):
        self.operations.clear()
        self._counter = 0
    
    def get_by_type(self, op_type: str) -> List[SemanticOperation]:
        return [op for op in self.operations if op.op_type == op_type]


class SemRewriter:
    def __init__(self, dialect: str = 'duckdb'):
        self.dialect = dialect
        self.pending = PendingOperations()
        self._alias_counter = 0
        self._semantic_columns = set()  # Track columns created by semantic operations
    
    def rewrite(self, sql: str, optimize_plan: bool = True) -> Tuple[str, List[SemanticOperation]]:
        self.pending.clear()
        self._alias_counter = 0
        self._semantic_columns.clear()  # Reset semantic columns for each rewrite
        self._sem_expr_to_alias = {}  # Track mapping from semantic expressions to aliases
        
        try:
            parsed_root = parse_one(sql, dialect=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL: {e}")

        root = parsed_root.copy() if hasattr(parsed_root, 'copy') else parsed_root
        
        self._extract_semantic_operations(root)
        
        if optimize_plan:
            root = self._apply_traditional_optimizations(root)
        
        needed_columns = self._collect_needed_columns(root)
        canonical_sql = self._emit_canonical_sql(root, needed_columns)
        
        return canonical_sql, self.pending.operations
    
    def _apply_traditional_optimizations(self, root: exp.Expression) -> exp.Expression:
        try:
            optimized = optimize(root, dialect=self.dialect, rules=[
                "normalize",
                "qualify_columns",
                "pushdown_predicates",
                "simplify",
                "merge_subqueries",
                "eliminate_joins"
            ])
            return optimized
        except Exception:
            return root
    
    def _extract_semantic_operations(self, root: exp.Expression):
        # First, recursively process any subqueries in FROM clause
        self._process_subqueries(root)
        
        self._rewrite_from_clause(root)  # Handle SEM_JOIN first
        self._rewrite_where(root)
        self._rewrite_group_by(root)
        self._rewrite_agg(root)
        self._rewrite_select_clause(root)
        self._rewrite_distinct(root)
        self._rewrite_order_by(root)
    
    def _process_subqueries(self, root: exp.Expression):
        """Recursively process semantic operations in subqueries."""
        from_clause = root.find(exp.From)
        if not from_clause:
            return
        
        # Check if FROM contains a subquery
        if isinstance(from_clause.this, exp.Subquery):
            subquery_select = from_clause.this.this
            if isinstance(subquery_select, exp.Select):
                # Recursively extract semantic operations from the subquery
                self._extract_semantic_operations(subquery_select)
    
    def _collect_needed_columns(self, root: exp.Expression) -> set:
        needed = set()
        
        for select in root.find_all(exp.Select):
            for expr in select.expressions:
                for col in expr.find_all(exp.Column):
                    # Exclude columns created by semantic operations
                    if col.name not in self._semantic_columns:
                        needed.add(col.name)
        
        for where in root.find_all(exp.Where):
            for col in where.find_all(exp.Column):
                if col.name not in self._semantic_columns:
                    needed.add(col.name)
        
        for group in root.find_all(exp.Group):
            for expr in group.expressions:
                for col in expr.find_all(exp.Column):
                    if col.name not in self._semantic_columns:
                        needed.add(col.name)
        
        for order in root.find_all(exp.Order):
            for expr in order.expressions:
                for col in expr.find_all(exp.Column):
                    if col.name not in self._semantic_columns:
                        needed.add(col.name)
        
        # Add columns referenced in semantic operations (via {column_name} placeholders)
        for op in self.pending.operations:
            for arg in op.args:
                # Extract column names from {column_name} patterns in prompts
                matches = re.findall(r'\{(\w+)\}', arg)
                for match in matches:
                    needed.add(match)
        
        return needed if needed else {'*'}
    
    def _rewrite_from_clause(self, root: exp.Expression):
        """Handle SEM_JOIN, SEM_INTERSECT, and SEM_EXCEPT in FROM clause."""
        from_clause = root.find(exp.From)
        if not from_clause:
            return
        
        # Check if FROM clause contains semantic operations
        table_expr = from_clause.this
        
        # These operations can be directly in FROM or wrapped in a Table with alias
        sem_expr = None
        table_alias = None
        op_type = None
        
        if isinstance(table_expr, exp.Anonymous) and table_expr.name:
            op_name = table_expr.name.upper()
            if op_name in ('SEM_JOIN', 'SEM_INTERSECT', 'SEM_INTERSECT_ALL', 'SEM_EXCEPT', 'SEM_EXCEPT_ALL'):
                sem_expr = table_expr
                op_type = op_name
        elif isinstance(table_expr, exp.Table) and hasattr(table_expr, 'this'):
            # Check if the table's this is a semantic operation
            if isinstance(table_expr.this, exp.Anonymous) and table_expr.this.name:
                op_name = table_expr.this.name.upper()
                if op_name in ('SEM_JOIN', 'SEM_INTERSECT', 'SEM_INTERSECT_ALL', 'SEM_EXCEPT', 'SEM_EXCEPT_ALL'):
                    sem_expr = table_expr.this
                    table_alias = table_expr.alias if hasattr(table_expr, 'alias') else None
                    op_type = op_name
        
        if sem_expr and op_type:
            # Extract arguments
            args = [arg.sql() for arg in sem_expr.expressions]
            
            # Map operation names to internal types
            if op_type == 'SEM_JOIN':
                self.pending.add('join', args)
            elif op_type == 'SEM_INTERSECT':
                self.pending.add('intersect', args)
            elif op_type == 'SEM_INTERSECT_ALL':
                self.pending.add('intersect_all', args)
            elif op_type == 'SEM_EXCEPT':
                self.pending.add('except', args)
            elif op_type == 'SEM_EXCEPT_ALL':
                self.pending.add('except_all', args)
            
            # Replace semantic operation with a placeholder table name
            # For JOIN, use the first table; for INTERSECT/EXCEPT, we'll handle differently
            if len(args) >= 2:
                first_arg = args[0].strip("'\"")
                # Create a simple table reference
                new_table = exp.Table(this=exp.Identifier(this=first_arg))
                if table_alias:
                    new_table.set("alias", table_alias)
                from_clause.set("this", new_table)
    
    def _rewrite_where(self, root: exp.Expression):
        where_clause = root.find(exp.Where)
        if not where_clause:
            return
        
        new_condition = self._strip_sem_where(where_clause.this)
        
        if self._is_true_literal(new_condition):
            where_clause.pop()
        else:
            where_clause.set("this", new_condition)
    
    def _strip_sem_where(self, bool_expr: exp.Expression) -> exp.Expression:
        if isinstance(bool_expr, exp.Anonymous) and bool_expr.name and bool_expr.name.upper() == 'SEM_WHERE':
            args = [arg.sql() for arg in bool_expr.expressions]
            self.pending.add('where', args)
            return exp.true()
        
        if isinstance(bool_expr, exp.And):
            left = self._strip_sem_where(bool_expr.left)
            right = self._strip_sem_where(bool_expr.right)
            
            if self._is_true_literal(left):
                return right
            if self._is_true_literal(right):
                return left
            
            return exp.And(this=left, expression=right)
        
        if isinstance(bool_expr, exp.Or):
            left = self._strip_sem_where(bool_expr.left)
            right = self._strip_sem_where(bool_expr.right)
            
            if self._is_true_literal(left) and self._is_true_literal(right):
                return exp.true()
            
            return exp.Or(this=left, expression=right)
        
        return bool_expr
    
    def _is_true_literal(self, expr: exp.Expression) -> bool:
        return isinstance(expr, exp.Boolean) and expr.this
    
    def _rewrite_group_by(self, root: exp.Expression):
        group_clause = root.find(exp.Group)
        if not group_clause:
            return
        
        for expr in group_clause.expressions:
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_GROUP_BY':
                args = [arg.sql() for arg in expr.expressions]
                self.pending.add('groupby', args)
    
    def _rewrite_order_by(self, root: exp.Expression):
        order_clause = root.find(exp.Order)
        if not order_clause:
            return
        
        for order_expr in order_clause.expressions:
            expr = order_expr.this if hasattr(order_expr, 'this') else order_expr
            
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_ORDER_BY':
                args = [arg.sql() for arg in expr.expressions]
                self.pending.add('orderby', args)
    
    def _rewrite_select_clause(self, root: exp.Expression):
        select = root.find(exp.Select)
        if not select:
            return
        
        # Track mapping from semantic expressions to their aliases
        self._sem_expr_to_alias = {}
        
        new_expressions = []
        for expr in select.expressions:
            if isinstance(expr, exp.Alias):
                inner = expr.this
                if isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_SELECT':
                    args = [arg.sql() for arg in inner.expressions]
                    args.append(f"'{expr.alias}'")
                    self.pending.add('select', args)
                    self._semantic_columns.add(expr.alias)  # Track semantic column
                    # Track the mapping from expression to alias
                    self._sem_expr_to_alias[inner.sql()] = expr.alias
                    new_expressions.append(exp.column(expr.alias))
                elif isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_DISTINCT':
                    args = [arg.sql() for arg in inner.expressions]
                    args.append(f"'{expr.alias}'")
                    self.pending.add('distinct', args)
                    self._semantic_columns.add(expr.alias)  # Track semantic column
                    new_expressions.append(exp.column(expr.alias))
                elif isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_GROUP_BY':
                    # Handle SEM_GROUP_BY in SELECT clause - replace with cluster_id
                    args = [arg.sql() for arg in inner.expressions]
                    # Don't add to pending here - it's already added in _rewrite_group_by
                    # Just replace with cluster_id column reference
                    new_expressions.append(exp.alias_(exp.column('cluster_id'), expr.alias))
                else:
                    new_expressions.append(expr)
            else:
                new_expressions.append(expr)
        
        select.set("expressions", new_expressions)
    
    def _rewrite_distinct(self, root: exp.Expression):
        select = root.find(exp.Select)
        if not select or not select.args.get('distinct'):
            return
        
        for expr in select.expressions:
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_DISTINCT':
                args = [arg.sql() for arg in expr.expressions]
                self.pending.add('distinct', args)
    
    def _rewrite_agg(self, root: exp.Expression):
        select = root.find(exp.Select)
        if not select:
            return
        
        new_expressions = []
        for expr in select.expressions:
            if isinstance(expr, exp.Alias):
                inner = expr.this
                if isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_AGG':
                    args = [arg.sql() for arg in inner.expressions]
                    args.append(f"'{expr.alias}'")
                    self.pending.add('agg', args)
                    self._semantic_columns.add(expr.alias)  # Track semantic column
                    new_expressions.append(exp.column(expr.alias))
                else:
                    new_expressions.append(expr)
            else:
                new_expressions.append(expr)
        
        select.set("expressions", new_expressions)
    
    def _emit_canonical_sql(self, root: exp.Expression, needed_columns: set) -> str:
        lines = []
        
        child_sql = self._build_child_cte(root, needed_columns)
        lines.append(f"WITH _child AS ({child_sql})")
        
        current_relation = "_child"
        step = 1
        
        for op in self.pending.operations:
            next_relation = f"_sem_{step}"
            lines.append(f", {next_relation} AS (/* {op.op_type.upper()} operation applied via backend */)")
            current_relation = next_relation
            step += 1
        
        final_projection = self._extract_final_projection(root)
        
        # Check if SELECT has DISTINCT (non-semantic)
        select = root.find(exp.Select)
        distinct_keyword = ""
        if select and select.args.get('distinct'):
            # Check if it's semantic distinct
            has_sem_distinct = any(
                isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_DISTINCT'
                for expr in select.expressions
            )
            if not has_sem_distinct:
                distinct_keyword = "DISTINCT "
        
        final_query = f"SELECT {distinct_keyword}{final_projection} FROM {current_relation}"
        
        # Preserve traditional JOINs (after semantic join)
        # Need to replace any reference to the semantic join alias with current_relation
        joins = list(root.find_all(exp.Join))
        if joins:
            # Check if we have a semantic join operation
            has_sem_join = any(op.op_type == 'join' for op in self.pending.operations)
            if has_sem_join:
                # Get the alias used in the original FROM clause
                from_clause = root.find(exp.From)
                if from_clause and isinstance(from_clause.this, exp.Table):
                    original_alias = from_clause.this.alias if hasattr(from_clause.this, 'alias') else None
                    if original_alias:
                        # Replace the alias in JOIN conditions
                        for join in joins:
                            join_sql = join.sql()
                            # Replace alias references with current_relation
                            join_sql = join_sql.replace(f"{original_alias}.", f"{current_relation}.")
                            final_query += f" {join_sql}"
                    else:
                        for join in joins:
                            final_query += f" {join.sql()}"
                else:
                    for join in joins:
                        final_query += f" {join.sql()}"
            else:
                for join in joins:
                    final_query += f" {join.sql()}"
        
        # Preserve traditional GROUP BY (non-semantic grouping)
        group_clause = root.find(exp.Group)
        if group_clause:
            # Check if GROUP BY contains semantic operations
            has_semantic_group = any(
                isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_GROUP_BY'
                for expr in group_clause.expressions
            )
            
            # Check if we have semantic aggregation - if so, skip GROUP BY in final query
            # because SEM_AGG with group_by already produces aggregated results
            has_semantic_agg = any(op.op_type == 'agg' for op in self.pending.operations)
            
            if has_semantic_group and not has_semantic_agg:
                # Only add GROUP BY if there's no aggregation
                # (aggregation handles grouping internally)
                final_query += " GROUP BY cluster_id"
            elif not has_semantic_group:
                # Traditional GROUP BY (no semantic operations)
                # But check if any GROUP BY expression matches a semantic SELECT that's been processed
                group_by_exprs = []
                for expr in group_clause.expressions:
                    expr_sql = expr.sql()
                    # Check if this expression matches a semantic operation in our mapping
                    if expr_sql in self._sem_expr_to_alias:
                        # Use the alias instead of the semantic expression
                        matched_alias = self._sem_expr_to_alias[expr_sql]
                        group_by_exprs.append(matched_alias)
                    else:
                        group_by_exprs.append(expr_sql)
                
                final_query += f" GROUP BY {', '.join(group_by_exprs)}"
        
        # Preserve HAVING clause
        having_clause = root.find(exp.Having)
        if having_clause:
            final_query += f" HAVING {having_clause.this.sql()}"
        
        # Preserve traditional ORDER BY (non-semantic ordering)
        order_clause = root.find(exp.Order)
        if order_clause:
            # Check if ORDER BY contains semantic operations
            has_semantic_order = any(
                isinstance(expr.this if hasattr(expr, 'this') else expr, exp.Anonymous) and
                (expr.this if hasattr(expr, 'this') else expr).name and
                (expr.this if hasattr(expr, 'this') else expr).name.upper() == 'SEM_ORDER_BY'
                for expr in order_clause.expressions
            )
            
            if not has_semantic_order:
                order_by_sql = order_clause.sql().replace('ORDER BY', '').strip()
                final_query += f" ORDER BY {order_by_sql}"
        
        # Preserve LIMIT
        limit_clause = root.find(exp.Limit)
        if limit_clause:
            final_query += f" LIMIT {limit_clause.expression.sql()}"
        
        # Preserve OFFSET
        offset_clause = root.find(exp.Offset)
        if offset_clause:
            final_query += f" OFFSET {offset_clause.expression.sql()}"
        
        lines.append(final_query)
        
        return "\n".join(lines)
    
    def _build_child_cte(self, root: exp.Expression, needed_columns: set) -> str:
        from_clause = root.find(exp.From)
        if not from_clause:
            raise ValueError("Query must have a FROM clause")
        
        # Check if we have semantic set operations (JOIN, INTERSECT, EXCEPT) - if so, return placeholder
        has_set_operation = any(op.op_type in ('join', 'intersect', 'intersect_all', 'except', 'except_all') 
                                for op in self.pending.operations)
        if has_set_operation:
            # For semantic set operations, we return a placeholder that won't be executed
            return "SELECT 1 AS placeholder"
        
        table_expr = from_clause.this
        
        # Check if FROM clause contains a subquery (starts with SELECT)
        # In this case, the subquery already has its own WHERE clause
        table_sql = table_expr.sql()
        is_subquery = isinstance(table_expr, exp.Subquery) or (isinstance(table_expr, exp.Table) and table_sql.strip().startswith('('))
        
        if is_subquery:
            # For subqueries, we can only select columns available in its output
            # Use * to select all available columns from the subquery result
            base_sql = f"SELECT * FROM {table_sql}"
        else:
            # For simple tables, build SELECT with requested columns
            if needed_columns:
                col_list = ', '.join(sorted(needed_columns))
            else:
                col_list = '*'
            
            base_sql = f"SELECT {col_list} FROM {table_sql}"
            
            where_clause = root.find(exp.Where)
            if where_clause and not self._is_true_literal(where_clause.this):
                base_sql += f" WHERE {where_clause.this.sql()}"
        
        return base_sql
    
    def _extract_final_projection(self, root: exp.Expression) -> str:
        select = root.find(exp.Select)
        if not select:
            return "*"
        
        projections = []
        for expr in select.expressions:
            if isinstance(expr, exp.Alias):
                # Check if the alias is a semantic column that doesn't exist yet
                if expr.alias in self._semantic_columns:
                    # For semantic columns, just reference them directly
                    projections.append(expr.alias)
                else:
                    projections.append(f"{expr.this.sql()} AS {expr.alias}")
            elif isinstance(expr, exp.Column):
                # Check if this is a reference to a semantic column
                if expr.name in self._semantic_columns:
                    # Just use the column name directly
                    projections.append(expr.name)
                else:
                    projections.append(expr.sql())
            else:
                projections.append(expr.sql())
        
        return ', '.join(projections) if projections else '*'
    
    def explain(self) -> str:
        lines = []
        lines.append("OPTIMIZATION STRATEGY:")
        lines.append("  RULE: Traditional SQL operations execute BEFORE semantic operations")
        lines.append("  WHY: Traditional ops are cheap (μs), semantic ops are expensive (seconds)")
        lines.append("  HOW: Traditional predicates in base CTE, semantic ops in pipeline CTEs")
        lines.append("")
        
        lines.append("--- Detected Semantic Operations ---")
        if not self.pending.operations:
            lines.append("No semantic operations detected.")
        else:
            for i, op in enumerate(self.pending.operations, 1):
                lines.append(f"{i}. {op.op_type.upper()}")
                lines.append(f"   Args: {op.args}")
        
        if self.pending.operations:
            lines.append("")
            lines.append("--- Execution Plan ---")
            lines.append("Phase 1: Traditional SQL (optimized by sqlglot)")
            lines.append("  - Traditional WHERE predicates (cheap filtering)")
            lines.append("  - Traditional JOINs (optimized join order)")
            lines.append("  - Traditional projections (column pruning)")
            lines.append("")
            lines.append("Phase 2: Semantic Operations (on filtered data)")
            for op_type in ['join', 'intersect', 'intersect_all', 'except', 'except_all', 'where', 'select', 'groupby', 'orderby', 'distinct', 'agg']:
                ops = [op for op in self.pending.operations if op.op_type == op_type]
                if ops:
                    lines.append(f"  - SEM_{op_type.upper()} ({len(ops)} operation(s))")
        
        return "\n".join(lines)

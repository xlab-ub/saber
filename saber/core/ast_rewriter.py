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
        
        # Check if query has CTEs - if so, process each CTE separately
        if isinstance(root, exp.Select) and root.ctes:
            # Process CTEs recursively
            for cte in root.ctes:
                self._extract_semantic_operations(cte.this)
        
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
    
    def _extract_nested_semantic_ops(self, expression: exp.Expression) -> exp.Expression:
        """
        Recursively find and extract semantic operations nested within an expression.
        Replaces the semantic operation node with a column reference to its generated alias.
        """
        # logging.warning(f"DEBUG: Extracting nested ops from: {expression.sql()}")
        def transformer(node):
            if isinstance(node, exp.Anonymous) and node.name:
                op_name = node.name.upper()
                if op_name in ('SEM_SELECT', 'SEM_AGG', 'SEM_DISTINCT'):
                    # logging.warning(f"DEBUG: Found nested {op_name}")
                    # Found a nested semantic op
                    args = [arg.sql(dialect=self.dialect) for arg in node.expressions]
                    
                    # Generate alias
                    alias = f"sem_extracted_{self._alias_counter}"
                    self._alias_counter += 1
                    
                    # Add to pending
                    if op_name == 'SEM_SELECT':
                        args.append(f"'{alias}'")
                        self.pending.add('select', args)
                    elif op_name == 'SEM_AGG':
                        args.append(f"'{alias}'")
                        self.pending.add('agg', args)
                    elif op_name == 'SEM_DISTINCT':
                        args.append(f"'{alias}'")
                        self.pending.add('distinct', args)
                        
                    self._semantic_columns.add(alias)
                    
                    # Return the replacement node
                    return exp.Column(this=exp.Identifier(this=alias, quoted=False))
            return node

        return expression.transform(transformer)

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
        # Helper to process a node and its children
        def visit(node):
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    self._extract_semantic_operations(subquery_select)
            
            # Check for subqueries in other clauses (WHERE, HAVING, SELECT list)
            for child in node.args.values():
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, exp.Expression):
                            visit(item)
                elif isinstance(child, exp.Expression):
                    visit(child)

        # Start traversal from root's children to avoid infinite recursion if root is a Subquery
        # (though _extract_semantic_operations calls this, so we should be careful)
        
        # Check FROM clause specifically first (common case)
        from_clause = root.find(exp.From)
        if from_clause:
            if isinstance(from_clause.this, exp.Subquery):
                visit(from_clause.this)
        
        # Check WHERE clause for subqueries
        where_clause = root.find(exp.Where)
        if where_clause:
            visit(where_clause)
            
        # Check HAVING clause
        having_clause = root.find(exp.Having)
        if having_clause:
            visit(having_clause)
            
        # Check SELECT list for subqueries
        if isinstance(root, exp.Select):
            for expr in root.expressions:
                visit(expr)
    
    def _collect_needed_columns(self, root: exp.Expression) -> set:
        needed = set()
        
        for select in root.find_all(exp.Select):
            for expr in select.expressions:
                # Skip aggregate functions and their aliases - they need to be computed in the query
                if isinstance(expr, exp.Alias):
                    inner = expr.this
                    # Check if this is an aggregate function
                    if isinstance(inner, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
                        # For aggregates, collect the columns referenced inside them
                        for col in inner.find_all(exp.Column):
                            if col.name not in self._semantic_columns:
                                needed.add(col.name)
                        continue
                
                # For other expressions, collect columns
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
        
        # Add columns referenced in semantic operations (via {column_name} or {alias.column} placeholders)
        for op in self.pending.operations:
            for arg in op.args:
                # Extract column names from {column_name} or {alias.column} patterns in prompts
                # Pattern matches: {column}, {alias.column}, {column:left}, {column:right}
                matches = re.findall(r'\{([^}]+)\}', arg)
                for match in matches:
                    # Strip whitespace and handle different formats
                    clean_match = match.strip()
                    # Remove :left or :right suffix if present (for JOIN operations)
                    if ':' in clean_match:
                        clean_match = clean_match.split(':')[0].strip()
                    # If it's alias.column format, extract the column part
                    if '.' in clean_match:
                        column_name = clean_match.split('.')[-1].strip()
                        needed.add(column_name)
                    else:
                        needed.add(clean_match)
        
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
            args = [arg.sql(dialect=self.dialect) for arg in sem_expr.expressions]
            
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
        
        # Recursively extract nested semantic operations from the condition
        # e.g. WHERE SEM_SELECT(...) > 5
        new_condition = self._extract_nested_semantic_ops(new_condition)
        
        if self._is_true_literal(new_condition):
            where_clause.pop()
        else:
            where_clause.set("this", new_condition)
    
    def _strip_sem_where(self, bool_expr: exp.Expression) -> exp.Expression:
        if isinstance(bool_expr, exp.Anonymous) and bool_expr.name and bool_expr.name.upper() == 'SEM_WHERE':
            args = [arg.sql(dialect=self.dialect) for arg in bool_expr.expressions]
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
        
        new_expressions = []
        for expr in group_clause.expressions:
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_GROUP_BY':
                args = [arg.sql(dialect=self.dialect) for arg in expr.expressions]
                self.pending.add('groupby', args)
                new_expressions.append(expr) # Keep it, _emit_canonical_sql will handle skipping
            else:
                # Handle nested semantic ops in GROUP BY
                # e.g. GROUP BY LENGTH(SEM_SELECT(...))
                transformed = self._extract_nested_semantic_ops(expr)
                new_expressions.append(transformed)
        
        group_clause.set("expressions", new_expressions)
    
    def _rewrite_order_by(self, root: exp.Expression):
        order_clause = root.find(exp.Order)
        if not order_clause:
            return
        
        new_expressions = []
        for order_expr in order_clause.expressions:
            expr = order_expr.this if hasattr(order_expr, 'this') else order_expr
            
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_ORDER_BY':
                args = [arg.sql(dialect=self.dialect) for arg in expr.expressions]
                self.pending.add('orderby', args)
                # Remove from ORDER BY since it's handled semantically
                continue
            
            # Check for SEM_SELECT in ORDER BY (e.g. ORDER BY SEM_SELECT(...) DESC)
            # This happens when the user wants to order by a semantic extraction without selecting it
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_SELECT':
                # Extract as a semantic operation
                args = [arg.sql(dialect=self.dialect) for arg in expr.expressions]
                
                # Generate a unique alias for this semantic column
                alias = f"sem_order_{self._alias_counter}"
                self._alias_counter += 1
                args.append(f"'{alias}'")
                
                self.pending.add('select', args)
                self._semantic_columns.add(alias)
                
                # Replace the expression in ORDER BY with the alias
                # We need to preserve the ordering direction (ASC/DESC)
                if isinstance(order_expr, exp.Ordered):
                    new_expressions.append(exp.Ordered(this=exp.Column(this=exp.Identifier(this=alias, quoted=False)), desc=order_expr.args.get('desc')))
                else:
                    new_expressions.append(exp.Column(this=exp.Identifier(this=alias, quoted=False)))
                continue
            
            # Handle nested semantic ops in ORDER BY
            # e.g. ORDER BY LENGTH(SEM_SELECT(...))
            transformed = self._extract_nested_semantic_ops(order_expr)
            new_expressions.append(transformed)
            
        if not new_expressions:
            # All order by expressions were semantic, remove the clause
            # (It will be re-added as semantic operation)
            root.set("order", None)
        else:
            order_clause.set("expressions", new_expressions)
    
    def _rewrite_select_clause(self, root: exp.Expression):
        select = root.find(exp.Select)
        if not select:
            return
        
        # Track mapping from semantic expressions to their aliases
        self._sem_expr_to_alias = {}
        
        # logging.warning("DEBUG: Entering _rewrite_select_clause")
        new_expressions = []
        for expr in select.expressions:
            if isinstance(expr, exp.Alias):
                inner = expr.this
                if isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_SELECT':
                    args = [arg.sql(dialect=self.dialect) for arg in inner.expressions]
                    args.append(f"'{expr.alias}'")
                    self.pending.add('select', args)
                    self._semantic_columns.add(expr.alias)  # Track semantic column
                    # Track the mapping from expression to alias
                    self._sem_expr_to_alias[inner.sql(dialect=self.dialect)] = expr.alias
                    new_expressions.append(exp.column(expr.alias))
                elif isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_DISTINCT':
                    args = [arg.sql(dialect=self.dialect) for arg in inner.expressions]
                    args.append(f"'{expr.alias}'")
                    self.pending.add('distinct', args)
                    self._semantic_columns.add(expr.alias)  # Track semantic column
                    new_expressions.append(exp.column(expr.alias))
                elif isinstance(inner, exp.Anonymous) and inner.name and inner.name.upper() == 'SEM_GROUP_BY':
                    # Handle SEM_GROUP_BY in SELECT clause - replace with cluster_id
                    args = [arg.sql(dialect=self.dialect) for arg in inner.expressions]
                    # Don't add to pending here - it's already added in _rewrite_group_by
                    # Just replace with cluster_id column reference
                    new_expressions.append(exp.alias_(exp.column('cluster_id'), expr.alias))
                else:
                    # Top-level alias but inner is not a simple SEM_SELECT/AGG/DISTINCT
                    # It might be Alias(Func(SEM_SELECT...))
                    # We need to process the inner expression
                    new_inner = self._extract_nested_semantic_ops(inner)
                    expr.set("this", new_inner)
                    new_expressions.append(expr)
            else:
                # Not an alias, check for nested semantic ops
                # e.g. SELECT LENGTH(SEM_SELECT(...))
                transformed = self._extract_nested_semantic_ops(expr)
                new_expressions.append(transformed)
        
        select.set("expressions", new_expressions)
    
    def _rewrite_distinct(self, root: exp.Expression):
        select = root.find(exp.Select)
        if not select or not select.args.get('distinct'):
            return
        
        for expr in select.expressions:
            if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_DISTINCT':
                args = [arg.sql(dialect=self.dialect) for arg in expr.expressions]
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
                    args = [arg.sql(dialect=self.dialect) for arg in inner.expressions]
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
        # If no semantic operations, just return the original SQL
        if not self.pending.operations:
            # Fix STRUCT generation before returning
            sql = root.sql(dialect=self.dialect)
            if 'STRUCT(' in sql.upper():
                # Simple regex fix for common STRUCT issues in MySQL
                # Replace STRUCT(col) with col
                sql = re.sub(r'STRUCT\s*\(([^,)]+)\)', r'\1', sql, flags=re.IGNORECASE)
            return sql
        
        lines = []
        
        # Check if the query has existing CTEs (WITH clause)
        has_user_ctes = isinstance(root, exp.Select) and root.ctes
        
        if has_user_ctes:
            # For queries with CTEs containing semantic operations:
            # 1. Use the CTE's base table as _child
            # 2. Apply semantic operations on _child
            # 3. User's main SELECT uses the transformed CTE
            
            # Get the first CTE that has semantic operations
            # For now, assume the pattern: WITH cte AS (SELECT ... WHERE SEM_WHERE(...))
            #                              SELECT ... FROM cte
            cte_with_sem_ops = root.ctes[0]  # Assume first CTE has semantic ops
            cte_name = cte_with_sem_ops.alias
            
            # Build _child from the CTE's FROM clause (base table)
            cte_select = cte_with_sem_ops.this
            cte_from = cte_select.find(exp.From)
            if cte_from:
                base_table = cte_from.this.sql(dialect=self.dialect)
                # Get columns from CTE SELECT (excluding semantic operations)
                col_list = '*'
                child_sql = f"SELECT {col_list} FROM {base_table}"
            else:
                child_sql = "SELECT 1 AS placeholder"
            
            lines.append(f"WITH _child AS ({child_sql})")
            current_relation = "_child"
            step = 1
        else:
            # No user CTEs - standard flow
            child_sql = self._build_child_cte(root, needed_columns)
            lines.append(f"WITH _child AS ({child_sql})")
            current_relation = "_child"
            step = 1
        
        for op in self.pending.operations:
            next_relation = f"_sem_{step}"
            lines.append(f", {next_relation} AS (/* {op.op_type.upper()} operation applied via backend */)")
            current_relation = next_relation
            step += 1
        
        # If we have user CTEs, we need to create a final CTE that matches the original CTE name
        if has_user_ctes:
            cte_name = root.ctes[0].alias
            cte_select = root.ctes[0].this
            # Extract the SELECT projection from the original CTE
            cte_projection = self._extract_projection_from_select(cte_select)
            lines.append(f", {cte_name} AS (SELECT {cte_projection} FROM {current_relation})")
            # Now the main query can reference the CTE name
        
        # For queries with user CTEs, extract the main SELECT projection
        if has_user_ctes:
            main_select = root  # The root is the main SELECT that uses the CTE
            final_projection = self._extract_final_projection(main_select)
            # Get the FROM clause to see which CTE it references
            main_from = main_select.find(exp.From)
            if main_from:
                cte_reference = main_from.this.sql(dialect=self.dialect)
                final_query_from = cte_reference  # Use the CTE name directly
            else:
                final_query_from = current_relation
        else:
            final_projection = self._extract_final_projection(root)
            final_query_from = current_relation
        
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
        
        # Check if we have aggregation in the original query
        select = root.find(exp.Select)
        has_aggregates = False
        if select:
            for expr in select.expressions:
                if isinstance(expr, exp.Alias) and isinstance(expr.this, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
                    has_aggregates = True
                    break
        
        # For queries with aggregation, we need to preserve the full query structure
        if has_aggregates and len(self.pending.operations) == 0:
            # No semantic operations, but has aggregation - use the original query structure
            # Extract the full SELECT with aggregations
            final_projection = self._extract_final_projection(root)
        
        # Use the appropriate FROM clause
        if has_user_ctes:
            final_query = f"SELECT {distinct_keyword}{final_projection} FROM {final_query_from}"
        else:
            final_query = f"SELECT {distinct_keyword}{final_projection} FROM {current_relation}"
        
        # Preserve traditional JOINs (after semantic join)
        # Note: JOINs are already included in the base CTE (_build_child_cte),
        # so we only add them here if we have semantic operations that modify the data
        joins = list(root.find_all(exp.Join))
        if joins:
            # Check if we have semantic operations - if yes, JOINs are already in base CTE
            # Only add JOINs to final query if there are NO semantic operations that would
            # transform the data (in which case we're selecting from _sem_N, not _child)
            has_semantic_ops = len(self.pending.operations) > 0
            has_sem_join = any(op.op_type == 'join' for op in self.pending.operations)
            
            if not has_semantic_ops:
                # No semantic operations - JOINs should be in base, but add them here too
                # Actually, if no semantic ops, we're selecting directly from _child which has JOINs
                # So we don't need to add them again
                pass
            elif has_sem_join:
                # Semantic join operation - handle specially
                from_clause = root.find(exp.From)
                if from_clause and isinstance(from_clause.this, exp.Table):
                    original_alias = from_clause.this.alias if hasattr(from_clause.this, 'alias') else None
                    if original_alias:
                        # Replace the alias in JOIN conditions
                        for join in joins:
                            join_sql = join.sql(dialect=self.dialect)
                            # Replace alias references with current_relation
                            join_sql = join_sql.replace(f"{original_alias}.", f"{current_relation}.")
                            final_query += f" {join_sql}"
                    else:
                        for join in joins:
                            final_query += f" {join.sql(dialect=self.dialect)}"
        
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
            
            # Check if SELECT has traditional aggregation functions (AVG, COUNT, SUM, etc.)
            select = root.find(exp.Select)
            has_traditional_agg = False
            if select:
                for expr in select.expressions:
                    # Check for aggregate functions like AVG, COUNT, SUM, MAX, MIN
                    if expr.find(exp.AggFunc):
                        has_traditional_agg = True
                        break
            
            if has_semantic_group and not has_semantic_agg:
                if has_traditional_agg:
                    # Query uses traditional aggregation (AVG, COUNT, etc.) with SEM_GROUP_BY
                    # Need GROUP BY cluster_id for proper SQL aggregation
                    final_query += " GROUP BY cluster_id"
                else:
                    # No aggregation - just adds cluster_id column, no SQL GROUP BY needed
                    # Adding GROUP BY would cause "column must appear in GROUP BY" errors with SELECT *
                    pass
            elif not has_semantic_group:
                # Traditional GROUP BY (no semantic operations)
                # But check if any GROUP BY expression matches a semantic SELECT that's been processed
                group_by_exprs = []
                for expr in group_clause.expressions:
                    expr_sql = expr.sql(dialect=self.dialect)
                    
                    # Check if this is a SEM_SELECT expression
                    if isinstance(expr, exp.Anonymous) and expr.name and expr.name.upper() == 'SEM_SELECT':
                        # This is a SEM_SELECT in GROUP BY - need to find its corresponding alias
                        # Look for the alias in the SELECT clause
                        select = root.find(exp.Select)
                        if select:
                            for select_expr in select.expressions:
                                if isinstance(select_expr, exp.Alias):
                                    # Check if the aliased expression matches this SEM_SELECT
                                    if isinstance(select_expr.this, exp.Anonymous) and select_expr.this.sql(dialect=self.dialect) == expr_sql:
                                        group_by_exprs.append(select_expr.alias)
                                        break
                            else:
                                # Couldn't find alias, just use the expression as-is
                                group_by_exprs.append(expr_sql)
                        else:
                            group_by_exprs.append(expr_sql)
                    # Check if this expression matches a semantic operation in our mapping
                    elif expr_sql in self._sem_expr_to_alias:
                        # Use the alias instead of the semantic expression
                        matched_alias = self._sem_expr_to_alias[expr_sql]
                        group_by_exprs.append(matched_alias)
                    else:
                        # Strip table aliases from GROUP BY expressions
                        group_by_exprs.append(self._strip_table_aliases(expr_sql))
                
                final_query += f" GROUP BY {', '.join(group_by_exprs)}"
        
        # Preserve HAVING clause
        having_clause = root.find(exp.Having)
        if having_clause:
            having_sql = having_clause.this.sql(dialect=self.dialect)
            # Strip table aliases from HAVING clause
            having_sql = self._strip_table_aliases(having_sql)
            final_query += f" HAVING {having_sql}"
        
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
                order_by_sql = order_clause.sql(dialect=self.dialect).replace('ORDER BY', '').strip()
                # Strip table aliases from ORDER BY clause
                order_by_sql = self._strip_table_aliases(order_by_sql)
                final_query += f" ORDER BY {order_by_sql}"
        
        # Preserve LIMIT
        limit_clause = root.find(exp.Limit)
        if limit_clause:
            final_query += f" LIMIT {limit_clause.expression.sql(dialect=self.dialect)}"
        
        # Preserve OFFSET
        offset_clause = root.find(exp.Offset)
        if offset_clause:
            final_query += f" OFFSET {offset_clause.expression.sql(dialect=self.dialect)}"
        
        lines.append(final_query)
        
        result_sql = "\n".join(lines)
        
        # Final cleanup for MySQL STRUCT artifacts
        if self.dialect == 'mysql' and 'STRUCT(' in result_sql.upper():
            # Replace STRUCT(col) with col
            result_sql = re.sub(r'STRUCT\s*\(([^,)]+)\)', r'\1', result_sql, flags=re.IGNORECASE)
            
        return result_sql
    
    def _build_child_cte(self, root: exp.Expression, needed_columns: set) -> str:
        # For queries with existing CTEs, we DON'T build a _child CTE
        # Instead, the semantic operations pipeline will reference the CTEs directly
        # This is handled in _emit_canonical_sql by preserving the CTEs
        if isinstance(root, exp.Select) and root.ctes:
            # Don't create _child - return placeholder since we'll use existing CTEs
            # The actual table to use will be determined in _emit_canonical_sql
            return "SELECT 1 AS placeholder"
        
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
        table_sql = table_expr.sql(dialect=self.dialect)
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
            
            # Preserve traditional JOINs in the base CTE
            joins = list(root.find_all(exp.Join))
            if joins:
                for join in joins:
                    base_sql += f" {join.sql(dialect=self.dialect)}"
            
            where_clause = root.find(exp.Where)
            if where_clause and not self._is_true_literal(where_clause.this):
                base_sql += f" WHERE {where_clause.this.sql(dialect=self.dialect)}"
        
        return base_sql
    
    def _extract_projection_from_select(self, select_expr: exp.Select) -> str:
        """Extract column projections from a SELECT, excluding semantic operations."""
        projections = []
        for expr in select_expr.expressions:
            if isinstance(expr, exp.Alias):
                # Keep the alias but check if it's semantic
                col_name = expr.alias
                if not isinstance(expr.this, exp.Anonymous):
                    # Not a semantic function - keep as is
                    projections.append(expr.sql(dialect=self.dialect))
                else:
                    # Semantic function - just reference the alias
                    projections.append(col_name)
            else:
                projections.append(expr.sql(dialect=self.dialect))
        return ', '.join(projections) if projections else '*'
    
    def _strip_table_aliases(self, sql_fragment: str) -> str:
        """Strip table alias prefixes from column references in SQL fragment.
        
        Converts: m.title, d.name -> title, name
        Preserves: function calls, literals, operators
        """
        # Pattern to match table_alias.column_name
        # Match word boundaries to avoid partial matches
        pattern = r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b'
        
        def replace_match(match):
            # Just return the column name without the table prefix
            return match.group(2)
        
        return re.sub(pattern, replace_match, sql_fragment)
    
    def _extract_final_projection(self, root: exp.Expression) -> str:
        select = root.find(exp.Select)
        if not select:
            return '*'
        
        projections = []
        for expr in select.expressions:
            if isinstance(expr, exp.Alias):
                # Check if the alias is a semantic column that doesn't exist yet
                if expr.alias in self._semantic_columns:
                    # For semantic columns created by operations like SEM_SELECT,
                    # the backend will add the column to the DataFrame, so we can
                    # just reference the alias name directly in the final projection
                    projections.append(expr.alias)
                else:
                    # For non-semantic aliases, strip table prefix from column references
                    # since after JOIN flattening, columns don't have table prefixes
                    inner_expr = expr.this
                    if isinstance(inner_expr, exp.Column):
                        # Strip table prefix: m.title -> title
                        col_name = inner_expr.name if hasattr(inner_expr, 'name') else str(inner_expr)
                        projections.append(f"{col_name} AS {expr.alias}")
                    else:
                        projections.append(f"{inner_expr.sql(dialect=self.dialect)} AS {expr.alias}")
            elif isinstance(expr, exp.Column):
                # Strip table prefix from column references
                # After JOIN and semantic operations, columns are flattened without table prefixes
                col_name = expr.name if hasattr(expr, 'name') else str(expr)
                projections.append(col_name)
            else:
                projections.append(expr.sql(dialect=self.dialect))
        
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

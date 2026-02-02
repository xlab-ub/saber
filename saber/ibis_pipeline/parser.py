"""
SQL Parser for Ibis Pipeline.

Parses SQL with SEM_* operations into a LogicalPlan with first-class
semantic operator nodes.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from sqlglot import parse_one, exp
from sqlglot.errors import ParseError

from .logical_plan import LogicalPlan, PlanNode, ScanNode, RelationalNode, SemanticNode
from .operators import (
    SemOp, SemFilter, SemFilterMarker, SemProject, SemJoin, SemGroupBy,
    SemAgg, SemDistinct, SemOrderBy, SemIntersect, SemExcept
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedSemOp:
    """Parsed semantic operation from SQL."""
    op_type: str
    args: List[str]
    alias: Optional[str] = None
    position: int = 0


class IbisSQLParser:
    """Parser that converts SQL with SEM_* to a LogicalPlan."""
    
    # Pattern for finding SEM_* function starts (we'll manually parse the arguments)
    SEM_START_PATTERN = re.compile(
        r"SEM_(WHERE|SELECT|JOIN|GROUP_BY|AGG|DISTINCT|ORDER_BY|INTERSECT(?:_ALL)?|EXCEPT(?:_ALL)?)"
        r"\s*\(",
        re.IGNORECASE | re.DOTALL
    )
    
    def __init__(self, dialect: str = 'duckdb'):
        self.dialect = dialect
        self._table_schemas: Dict[str, List[str]] = {}
    
    def set_table_schema(self, table_name: str, columns: List[str]):
        """Register table schema."""
        self._table_schemas[table_name] = columns
    
    def parse(self, sql: str) -> LogicalPlan:
        """Parse SQL into a LogicalPlan."""
        plan = LogicalPlan()
        for name, cols in self._table_schemas.items():
            plan.set_table_schema(name, cols)
        
        # Extract and temporarily replace SEM_* calls
        sem_ops, cleaned_sql = self._extract_sem_ops(sql)
        
        # Parse the cleaned SQL structure
        try:
            ast = parse_one(cleaned_sql, dialect=self.dialect)
        except ParseError as e:
            logger.warning(f"Failed to parse cleaned SQL: {e}")
            # Try original SQL
            ast = parse_one(sql, dialect=self.dialect)
        
        # Build plan from AST
        plan.root = self._build_plan_from_ast(ast, sem_ops, plan)
        
        return plan
    
    def _extract_sem_ops(self, sql: str) -> Tuple[List[ParsedSemOp], str]:
        """Extract SEM_* operations and return cleaned SQL."""
        sem_ops = []
        
        # Detect WHERE clause position to distinguish SEM_WHERE in WHERE vs SELECT
        where_clause_start = -1
        where_match = re.search(r'\bWHERE\b', sql, re.IGNORECASE)
        if where_match:
            where_clause_start = where_match.start()
        
        # Find all SEM_* calls using proper quote-aware parsing
        for match in self.SEM_START_PATTERN.finditer(sql):
            op_name = match.group(1).upper()
            start_pos = match.start()
            args_start = match.end()  # Position after the opening '('
            
            # Find the matching closing parenthesis, respecting quotes
            args_str, end_pos = self._extract_balanced_parens(sql, args_start)
            
            args = self._parse_args(args_str)
            
            # Look for AS alias after the function call
            alias = None
            alias_match = re.match(r'\s+AS\s+(\w+)', sql[end_pos:], re.IGNORECASE)
            if alias_match:
                alias = alias_match.group(1)
            
            # Determine if SEM_WHERE is in SELECT clause (for marking) vs WHERE clause (for filtering)
            op_type = self._normalize_op_type(op_name)
            if op_type == 'where':
                # Check if this SEM_WHERE appears before the WHERE clause (i.e., in SELECT)
                if where_clause_start == -1 or start_pos < where_clause_start:
                    op_type = 'where_marker'  # Mark rows, don't filter
            
            sem_ops.append(ParsedSemOp(
                op_type=op_type,
                args=args,
                alias=alias,
                position=start_pos
            ))
        
        # Create cleaned SQL by replacing SEM_* calls with placeholders
        cleaned_sql = sql
        
        # Replace SEM_WHERE in WHERE clause with TRUE
        # Only replace SEM_WHERE that's actually in the WHERE clause
        if where_clause_start >= 0:
            before_where = cleaned_sql[:where_clause_start]
            after_where = cleaned_sql[where_clause_start:]
            after_where = self._replace_sem_call(after_where, "SEM_WHERE", "TRUE")
            cleaned_sql = before_where + after_where
        
        # Replace SEM_WHERE in SELECT clause with placeholder column reference
        # This is for CASE WHEN SEM_WHERE(...) patterns
        cleaned_sql = self._replace_sem_call(cleaned_sql, "SEM_WHERE", "_sem_where_result")
        
        # Replace SEM_SELECT with a placeholder column
        # Pattern captures SEM_SELECT(...) and optional AS alias
        cleaned_sql = self._replace_sem_call(cleaned_sql, "SEM_SELECT", "{alias}")
        
        # Replace SEM_JOIN in FROM clause - needs special handling for table extraction
        # For now, use a simpler approach: just replace with a cross join placeholder
        cleaned_sql = self._replace_sem_join(cleaned_sql)
        
        # Replace SEM_GROUP_BY with column reference
        cleaned_sql = self._replace_sem_call(cleaned_sql, "SEM_GROUP_BY", "cluster_id")
        
        # Replace SEM_AGG - needs special handling for alias
        cleaned_sql = self._replace_sem_agg(cleaned_sql)
        
        # Replace SEM_DISTINCT
        cleaned_sql = self._replace_sem_call(cleaned_sql, "SEM_DISTINCT", "DISTINCT *")
        
        # Replace SEM_ORDER_BY
        cleaned_sql = self._replace_sem_call(cleaned_sql, "SEM_ORDER_BY", "1")
        
        # Replace SEM_INTERSECT and SEM_EXCEPT
        for set_op in ['INTERSECT_ALL', 'INTERSECT', 'EXCEPT_ALL', 'EXCEPT']:
            cleaned_sql = self._replace_sem_call(cleaned_sql, f"SEM_{set_op}", "(SELECT 1)")
        
        return sem_ops, cleaned_sql
    
    def _extract_balanced_parens(self, sql: str, start_pos: int) -> Tuple[str, int]:
        """Extract content between balanced parentheses, respecting quotes.
        
        Args:
            sql: The full SQL string
            start_pos: Position right after the opening '('
            
        Returns:
            Tuple of (content inside parens, position after closing ')')
        """
        depth = 1
        in_string = False
        string_char = None
        i = start_pos
        
        while i < len(sql) and depth > 0:
            char = sql[i]
            
            if char in ("'", '"') and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                # Check for escaped quote (two quotes in a row)
                if i + 1 < len(sql) and sql[i + 1] == string_char:
                    i += 1  # Skip the escaped quote
                else:
                    in_string = False
                    string_char = None
            elif not in_string:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
            
            i += 1
        
        # Content is from start_pos to i-1 (excluding the closing paren)
        content = sql[start_pos:i-1]
        return content, i
    
    def _replace_sem_call(self, sql: str, sem_name: str, replacement: str) -> str:
        """Replace SEM_* function calls with a replacement string, respecting quotes.
        
        Args:
            sql: The SQL string
            sem_name: Name like 'SEM_SELECT', 'SEM_WHERE', etc.
            replacement: The string to replace with
            
        Returns:
            Modified SQL string
        """
        result = []
        i = 0
        pattern = re.compile(rf"{sem_name}\s*\(", re.IGNORECASE)
        
        while i < len(sql):
            match = pattern.search(sql, i)
            if not match:
                result.append(sql[i:])
                break
            
            # Add everything before the match
            result.append(sql[i:match.start()])
            
            # Find the matching closing paren
            args_start = match.end()
            _, end_pos = self._extract_balanced_parens(sql, args_start)
            
            # Check for AS alias after the function call
            alias_match = re.match(r'\s+AS\s+(\w+)', sql[end_pos:], re.IGNORECASE)
            if alias_match and '{alias}' in replacement:
                alias = alias_match.group(1)
                result.append(replacement.replace('{alias}', alias))
                end_pos += alias_match.end()
            elif '{alias}' in replacement:
                result.append(replacement.replace('{alias}', 'sem_extracted'))
            else:
                result.append(replacement)
            
            i = end_pos
        
        return ''.join(result)
    
    def _replace_sem_join(self, sql: str) -> str:
        """Replace SEM_JOIN in FROM clause with CROSS JOIN, respecting quotes."""
        result = []
        i = 0
        pattern = re.compile(r"FROM\s+SEM_JOIN\s*\(", re.IGNORECASE)
        
        while i < len(sql):
            match = pattern.search(sql, i)
            if not match:
                result.append(sql[i:])
                break
            
            # Add everything before "SEM_JOIN"
            from_start = match.start()
            result.append(sql[i:from_start])
            result.append("FROM ")
            
            # Find the matching closing paren
            args_start = match.end()
            args_str, end_pos = self._extract_balanced_parens(sql, args_start)
            
            # Parse the arguments to get table names
            args = self._parse_args(args_str)
            if len(args) >= 2:
                left_table = args[0].strip("'\"")
                right_table = args[1].strip("'\"")
                result.append(f"{left_table} CROSS JOIN {right_table}")
            else:
                result.append("dual")  # Fallback
            
            i = end_pos
        
        return ''.join(result)
    
    def _replace_sem_agg(self, sql: str) -> str:
        """Replace SEM_AGG with NULL AS alias, respecting quotes."""
        result = []
        i = 0
        pattern = re.compile(r"SEM_AGG\s*\(", re.IGNORECASE)
        
        while i < len(sql):
            match = pattern.search(sql, i)
            if not match:
                result.append(sql[i:])
                break
            
            # Add everything before the match
            result.append(sql[i:match.start()])
            
            # Find the matching closing paren
            args_start = match.end()
            _, end_pos = self._extract_balanced_parens(sql, args_start)
            
            # Check for AS alias after the function call
            alias_match = re.match(r'\s+AS\s+(\w+)', sql[end_pos:], re.IGNORECASE)
            if alias_match:
                alias = alias_match.group(1)
                result.append(f"NULL AS {alias}")
                end_pos += alias_match.end()
            else:
                result.append("NULL AS agg_result")
            
            i = end_pos
        
        return ''.join(result)

    def _parse_args(self, args_str: str) -> List[str]:
        """Parse function arguments handling nested quotes and parens."""
        args = []
        current = []
        depth = 0
        in_string = False
        string_char = None
        
        for char in args_str:
            if char in ("'", '"') and not in_string:
                in_string = True
                string_char = char
                current.append(char)
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                current.append(char)
            elif char == '(' and not in_string:
                depth += 1
                current.append(char)
            elif char == ')' and not in_string:
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0 and not in_string:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            args.append(''.join(current).strip())
        
        return [a for a in args if a]
    
    def _normalize_op_type(self, op_name: str) -> str:
        """Normalize operation type name."""
        mapping = {
            'WHERE': 'where',
            'SELECT': 'select',
            'JOIN': 'join',
            'GROUP_BY': 'groupby',
            'AGG': 'agg',
            'DISTINCT': 'distinct',
            'ORDER_BY': 'orderby',
            'INTERSECT': 'intersect',
            'INTERSECT_ALL': 'intersect_all',
            'EXCEPT': 'except',
            'EXCEPT_ALL': 'except_all',
        }
        return mapping.get(op_name.upper(), op_name.lower())
    
    def _build_plan_from_ast(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                            plan: LogicalPlan) -> PlanNode:
        """Build logical plan from parsed AST and semantic ops."""
        
        # Start with scan nodes
        current_node = self._build_from_clause(ast, sem_ops, plan)
        
        # Add WHERE (including SEM_WHERE)
        current_node = self._build_where_clause(ast, sem_ops, plan, current_node)
        
        # Add GROUP BY (including SEM_GROUP_BY)
        current_node = self._build_group_by(ast, sem_ops, plan, current_node)
        
        # Add SELECT (including SEM_SELECT and SEM_AGG)
        current_node = self._build_select_clause(ast, sem_ops, plan, current_node)
        
        # Add DISTINCT (including SEM_DISTINCT)
        current_node = self._build_distinct(ast, sem_ops, plan, current_node)
        
        # Add ORDER BY (including SEM_ORDER_BY)
        current_node = self._build_order_by(ast, sem_ops, plan, current_node)
        
        # Add LIMIT
        current_node = self._build_limit(ast, plan, current_node)
        
        return current_node
    
    def _build_from_clause(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                          plan: LogicalPlan) -> PlanNode:
        """Build scan/join nodes from FROM clause."""
        
        # Check for SEM_JOIN
        join_ops = [op for op in sem_ops if op.op_type == 'join']
        if join_ops:
            op = join_ops[0]
            left_table = op.args[0].strip("'\"")
            right_table = op.args[1].strip("'\"")
            prompt = op.args[2].strip("'") if len(op.args) > 2 else ""
            backend = op.args[3].strip("'") if len(op.args) > 3 else "lotus"
            
            left_scan = plan.create_scan(left_table)
            right_scan = plan.create_scan(right_table)
            
            sem_join = SemJoin(
                prompt=prompt,
                left_table=left_table,
                right_table=right_table,
                backend=backend
            )
            
            return plan.create_semantic(sem_join, left_scan, right_scan)
        
        # Check for set operations (INTERSECT/EXCEPT)
        set_ops = [op for op in sem_ops if op.op_type in ('intersect', 'intersect_all', 'except', 'except_all')]
        if set_ops:
            op = set_ops[0]
            # Parse the two subqueries
            left_query = op.args[0].strip("'\"") if op.args else ""
            right_query = op.args[1].strip("'\"") if len(op.args) > 1 else ""
            
            # Create placeholder nodes - actual execution will evaluate subqueries
            left_scan = ScanNode(node_id=plan._next_node_id(), table_name=f"__subquery_left_{left_query}")
            right_scan = ScanNode(node_id=plan._next_node_id(), table_name=f"__subquery_right_{right_query}")
            
            if 'intersect' in op.op_type:
                sem_op = SemIntersect(is_all='all' in op.op_type)
            else:
                sem_op = SemExcept(is_all='all' in op.op_type)
            
            return plan.create_semantic(sem_op, left_scan, right_scan)
        
        # Extract tables from FROM clause
        from_clause = ast.find(exp.From) if isinstance(ast, exp.Select) else None
        if from_clause:
            table_expr = from_clause.this
            if isinstance(table_expr, exp.Table):
                table_name = table_expr.name
                alias = table_expr.alias if hasattr(table_expr, 'alias') else None
                return plan.create_scan(table_name, alias)
            elif isinstance(table_expr, exp.Subquery):
                # Handle subquery in FROM clause
                subquery_ast = table_expr.this  # The inner SELECT
                subquery_alias = table_expr.alias
                
                # Extract SEM_* ops from the subquery SQL
                subquery_sql = subquery_ast.sql(dialect=self.dialect)
                subquery_sem_ops = [op for op in sem_ops if self._op_in_sql(op, subquery_sql)]
                
                # Recursively build the plan for the subquery
                subquery_root = self._build_plan_from_ast(subquery_ast, subquery_sem_ops, plan)
                
                # Mark this as a subquery result with alias for proper handling
                if subquery_root and subquery_alias:
                    subquery_root.alias = subquery_alias
                
                return subquery_root
        
        # Fallback: create empty scan
        return ScanNode(node_id=plan._next_node_id())
    
    def _op_in_sql(self, op: ParsedSemOp, sql: str) -> bool:
        """Check if a semantic operation appears in the given SQL string."""
        # Simple heuristic: check if the operation's prompt appears in the SQL
        if op.args:
            # Check if the first argument (usually the prompt) appears in the SQL
            first_arg = op.args[0].strip("'\"")
            if first_arg in sql:
                return True
        return False
    
    def _build_where_clause(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                           plan: LogicalPlan, child: PlanNode) -> PlanNode:
        """Build filter nodes from WHERE clause."""
        current = child
        
        # Add SEM_WHERE_MARKER operations (SEM_WHERE in SELECT clause - adds boolean column)
        marker_ops = [op for op in sem_ops if op.op_type == 'where_marker']
        for op in marker_ops:
            prompt = op.args[0].strip("'") if op.args else ""
            backend = op.args[1].strip("'") if len(op.args) > 1 else "lotus"
            
            sem_marker = SemFilterMarker(prompt=prompt, backend=backend)
            current = plan.create_semantic(sem_marker, current)
        
        # Add SEM_WHERE operations (actual filtering in WHERE clause)
        where_ops = [op for op in sem_ops if op.op_type == 'where']
        for op in where_ops:
            prompt = op.args[0].strip("'") if op.args else ""
            backend = op.args[1].strip("'") if len(op.args) > 1 else "lotus"
            
            sem_filter = SemFilter(prompt=prompt, backend=backend)
            current = plan.create_semantic(sem_filter, current)
        
        # Add regular WHERE predicates
        # Use args.get('where') to get direct WHERE clause, not traversing into subqueries
        where_clause = ast.args.get('where') if isinstance(ast, exp.Select) else None
        if where_clause and where_clause.this:
            pred_sql = where_clause.this.sql(dialect=self.dialect)
            # Skip if it's just TRUE (our placeholder)
            if pred_sql.upper() != 'TRUE':
                current = plan.create_relational('filter', current, predicate=pred_sql)
        
        return current
    
    def _build_group_by(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                       plan: LogicalPlan, child: PlanNode) -> PlanNode:
        """Build group by nodes."""
        current = child
        
        # Add SEM_GROUP_BY operations
        groupby_ops = [op for op in sem_ops if op.op_type == 'groupby']
        for op in groupby_ops:
            column = op.args[0].strip("'") if op.args else "*"
            k = int(op.args[1]) if len(op.args) > 1 else 8
            backend = op.args[2].strip("'") if len(op.args) > 2 else "lotus"
            
            sem_groupby = SemGroupBy(column=column, k=k, backend=backend)
            current = plan.create_semantic(sem_groupby, current)
        
        # Add regular GROUP BY
        if isinstance(ast, exp.Select):
            group_clause = ast.args.get('group')
            if group_clause:
                group_cols = [e.sql(dialect=self.dialect) for e in group_clause.expressions]
                current = plan.create_relational('aggregate', current, group_by=group_cols)
        
        return current
    
    def _build_select_clause(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                            plan: LogicalPlan, child: PlanNode) -> PlanNode:
        """Build projection nodes from SELECT clause."""
        current = child
        
        # Add SEM_SELECT operations
        select_ops = [op for op in sem_ops if op.op_type == 'select']
        for op in select_ops:
            prompt = op.args[0].strip("'") if op.args else ""
            backend = op.args[1].strip("'") if len(op.args) > 1 else "lotus"
            alias = op.alias or "extracted"
            
            sem_project = SemProject(prompt=prompt, alias=alias, backend=backend)
            current = plan.create_semantic(sem_project, current)
        
        # Add SEM_AGG operations
        agg_ops = [op for op in sem_ops if op.op_type == 'agg']
        for op in agg_ops:
            # Get alias from op.alias if captured, or from args
            alias = op.alias or "agg_result"
            
            if len(op.args) >= 3:
                # Could be (column, prompt, backend) or (prompt, backend, something)
                # Check if first arg looks like a column name (no quotes, no braces)
                first_arg = op.args[0].strip("'\"")
                if "'" not in op.args[0] and '"' not in op.args[0] and '{' not in op.args[0]:
                    # First arg is column: (column, prompt, backend)
                    column = first_arg
                    prompt = op.args[1].strip("'\"")
                    backend = op.args[2].strip("'\"")
                else:
                    # First arg is prompt
                    column = None
                    prompt = first_arg
                    backend = op.args[1].strip("'\"")
            elif len(op.args) >= 2:
                # (prompt, backend) - no column
                column = None
                prompt = op.args[0].strip("'\"")
                backend = op.args[1].strip("'\"")
            else:
                # Only prompt
                column = None
                prompt = op.args[0].strip("'\"") if op.args else ""
                backend = "lotus"
            
            sem_agg = SemAgg(prompt=prompt, alias=alias, column=column, backend=backend)
            current = plan.create_semantic(sem_agg, current)
        
        # Extract projections from SELECT list
        if isinstance(ast, exp.Select):
            projections = []
            for expr in ast.expressions:
                projections.append(expr.sql(dialect=self.dialect))
            if projections:
                current.output_columns = projections
        
        return current
    
    def _build_distinct(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                       plan: LogicalPlan, child: PlanNode) -> PlanNode:
        """Build distinct nodes."""
        current = child
        
        # Add SEM_DISTINCT operations
        distinct_ops = [op for op in sem_ops if op.op_type == 'distinct']
        for op in distinct_ops:
            column = op.args[0].strip("'") if op.args else "*"
            backend = op.args[1].strip("'") if len(op.args) > 1 else "lotus"
            alias = op.args[2].strip("'") if len(op.args) > 2 else None
            
            sem_distinct = SemDistinct(column=column, alias=alias, backend=backend)
            current = plan.create_semantic(sem_distinct, current)
        
        # Check for regular DISTINCT
        if isinstance(ast, exp.Select) and ast.args.get('distinct'):
            current = plan.create_relational('distinct', current)
        
        return current
    
    def _build_order_by(self, ast: exp.Expression, sem_ops: List[ParsedSemOp],
                       plan: LogicalPlan, child: PlanNode) -> PlanNode:
        """Build order by nodes."""
        current = child
        
        # Add SEM_ORDER_BY operations
        orderby_ops = [op for op in sem_ops if op.op_type == 'orderby']
        for op in orderby_ops:
            if len(op.args) == 3:
                # column, prompt, backend
                column = op.args[0].strip("'")
                prompt = op.args[1].strip("'")
                backend = op.args[2].strip("'")
            else:
                # prompt, backend
                column = None
                prompt = op.args[0].strip("'") if op.args else ""
                backend = op.args[1].strip("'") if len(op.args) > 1 else "lotus"
            
            sem_orderby = SemOrderBy(prompt=prompt, column=column, backend=backend)
            current = plan.create_semantic(sem_orderby, current)
        
        # Add regular ORDER BY
        if isinstance(ast, exp.Select):
            order = ast.args.get('order')
            if order:
                order_cols = [e.sql(dialect=self.dialect) for e in order.expressions]
                current = plan.create_relational('sort', current, order_by=order_cols)
        
        return current
    
    def _build_limit(self, ast: exp.Expression, plan: LogicalPlan,
                    child: PlanNode) -> PlanNode:
        """Build limit node."""
        if isinstance(ast, exp.Select):
            limit = ast.args.get('limit')
            if limit:
                # Extract limit value - handle different sqlglot structures
                limit_val = None
                try:
                    # Try direct expression access (sqlglot structure: Limit -> expression)
                    if hasattr(limit, 'expression') and limit.expression:
                        limit_val = int(str(limit.expression))
                    # Fallback: try this.this pattern
                    elif hasattr(limit, 'this') and limit.this:
                        val = limit.this
                        if hasattr(val, 'this'):
                            limit_val = int(str(val.this))
                        else:
                            limit_val = int(str(val))
                    # Last resort: parse from SQL representation
                    if limit_val is None:
                        import re
                        limit_sql = limit.sql(dialect=self.dialect)
                        match = re.search(r'(\d+)', limit_sql)
                        if match:
                            limit_val = int(match.group(1))
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to parse LIMIT value: {e}")
                    limit_val = None
                
                if limit_val and limit_val > 0:
                    logger.debug(f"Creating LIMIT node with limit={limit_val}")
                    return plan.create_relational('limit', child, limit=limit_val)
        return child

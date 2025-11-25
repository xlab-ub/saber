"""
SQL parser for extracting semantic operations and table information.
"""
import re
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SQLParser:
    """Parses SQL queries to extract semantic operations and metadata."""
    
    def __init__(self):
        self.patterns = {
            # e.g., SEM_JOIN(left_table, right_table, user_prompt) or SEM_JOIN(left_table, right_table, user_prompt, system
            'join': r"SEM_JOIN\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*'([^']+)'\s*(?:,\s*'([^']+)')?\s*\)",
            # e.g., SEM_WHERE(user_prompt) or SEM_WHERE(user_prompt, system)
            'where': r"SEM_WHERE\s*\(\s*'([^']+)'\s*(?:,\s*'([^']+)')?\s*\)",
            # e.g., SEM_GROUP_BY(column, number_of_groups(int)) or SEM_GROUP_BY(column, number_of_groups(int), system)
            'group': r"SEM_GROUP_BY\s*\(\s*([^,]+)\s*,\s*(\d+)\s*(?:,\s*'([^']+)')?\s*\)",
            # e.g., SEM_AGG(user_prompt) AS alias or SEM_AGG(user_prompt, system) AS alias
            #     , SEM_AGG(column, user_prompt, system) AS alias
            'agg': r"SEM_AGG\s*\(\s*(?:([^,']+)\s*,\s*)?'([^']+)'\s*(?:,\s*'([^']+)')?\s*\)\s+AS\s+(\w+)",
            # e.g., SEM_SELECT(user_prompt) AS alias or SEM_SELECT(user_prompt, system) AS alias
            'select': r"SEM_SELECT\s*\(\s*'([^']+)'\s*(?:,\s*'([^']+)')?\s*\)\s+AS\s+(\w+)",
            # e.g., SEM_DISTINCT(column) or SEM_DISTINCT(column, system)
            'distinct': r"SEM_DISTINCT\s*\(\s*([^,]+)\s*(?:,\s*'([^']+)')?\s*\)",
            # e.g., SEM_ORDER_BY(user_prompt) or SEM_ORDER_BY(user_prompt, system)
            #     , SEM_ORDER_BY(column, user_prompt, system)
            'order': r"SEM_ORDER_BY\s*\(\s*(?:([^,']+)\s*,\s*)?'([^']+)'\s*(?:,\s*'([^']+)')?\s*\)",
            # e.g., SEM_EXCEPT(q1, q2)
            'except': r"SEM_EXCEPT\s*\(\s*(.+?)\s*\)",
            # e.g., SEM_EXCEPT_ALL(q1, q2)
            'except_all': r"SEM_EXCEPT_ALL\s*\(\s*(.+?)\s*\)",
            # e.g., SEM_INTERSECT(q1, q2)
            'intersect': r"SEM_INTERSECT\s*\(\s*(.+?)\s*\)",
            # e.g., SEM_INTERSECT_ALL(q1, q2)
            'intersect_all': r"SEM_INTERSECT_ALL\s*\(\s*(.+?)\s*\)",
        }
        
        # Updated patterns to handle quoted identifiers (backticks, double quotes, single quotes)
        # Match either: unquoted word OR `quoted name` OR "quoted name" OR 'quoted name'
        self.from_pattern = r"FROM\s+(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|(\w+))"
        self.from_join_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)(?:\s+AS)?\s+([a-zA-Z0-9_]+)'
        self.join_pattern = r'FROM\s+(\w+)(?:\s+AS\s+(\w+))?\s+JOIN\s+(\w+)(?:\s+AS\s+(\w+))?\s+ON\s+([^W\s][^H\s][^E\s][^R\s][^E\s]*?)(?=\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|\s*$)'
    
    def extract_table_aliases(self, sql: str) -> Dict[str, str]:
        """Extract table aliases from SQL query."""
        aliases = {}
        
        # Pattern that only matches when there's actually an explicit alias
        # FROM table_name AS alias_name or FROM table_name alias_name
        alias_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)\s+(?:AS\s+)?([a-zA-Z0-9_]+)(?=\s+(?:WHERE|ORDER|GROUP|HAVING|LIMIT|ON|\)|,|$))'
        
        for table, alias in re.findall(alias_pattern, sql, re.IGNORECASE):
            # Only add if table and alias are different (avoiding cases where WHERE/ORDER etc. are matched as aliases)
            if table.lower() != alias.lower() and alias.upper() not in ['WHERE', 'ORDER', 'GROUP', 'HAVING', 'LIMIT', 'ON']:
                aliases[alias] = table
                
        return aliases
    
    def extract_table_matches(self, sql: str) -> List[str]:
        """
        Extract table names from FROM clauses, handling quoted identifiers.
        
        Note: For unquoted multi-word table names (e.g., "FROM My Table"), only the first
        word will be extracted since SQL treats them as separate tokens. However, the
        db_adapter's table name normalization will still handle the full table name correctly
        when executing queries.
        """
        matches = re.findall(self.from_pattern, sql, re.IGNORECASE)
        # Flatten tuples from alternation groups - one of the 4 groups will match
        # Result is list of tuples like ('', '', '', 'table_name') or ('table name', '', '', '')
        result = []
        for match in matches:
            if isinstance(match, tuple):
                # Get the non-empty group
                table_name = next((m for m in match if m), None)
                if table_name:
                    result.append(table_name)
            else:
                result.append(match)
        return result
    
    def has_regular_join(self, sql: str) -> bool:
        """Check if SQL contains regular JOIN operations."""
        return bool(re.search(self.join_pattern, sql, re.IGNORECASE | re.DOTALL))
    
    def extract_regular_join_info(self, sql: str) -> Optional[Dict[str, str]]:
        """Extract information about regular JOIN operations."""
        match = re.search(self.join_pattern, sql, re.IGNORECASE | re.DOTALL)
        if not match:
            # Try a simpler pattern that captures everything after ON until the next clause
            simple_pattern = r'FROM\s+(\w+)(?:\s+AS\s+(\w+))?\s+JOIN\s+(\w+)(?:\s+AS\s+(\w+))?\s+ON\s+(.*?)(?=\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|\s*$)'
            match = re.search(simple_pattern, sql, re.IGNORECASE | re.DOTALL)
            
            if not match:
                return None
        
        return {
            'left_table': match.group(1),
            'left_alias': match.group(2) or match.group(1),
            'right_table': match.group(3),
            'right_alias': match.group(4) or match.group(3),
            'join_condition': match.group(5).strip()
        }
    
    def extract_semantic_operations(self, sql: str) -> Dict[str, List[Tuple]]:
        """Extract all semantic operations from SQL."""
        operations = {}
        
        # Extract semantic joins
        operations['join'] = []
        for match in re.finditer(self.patterns['join'], sql):
            left_table = match.group(1).strip()
            right_table = match.group(2).strip()
            user_prompt = match.group(3)
            system = match.group(4) if match.group(4) else 'lotus'
            operations['join'].append((left_table, right_table, user_prompt, system))
        
        # Extract semantic where clauses
        operations['where'] = []
        for match in re.finditer(self.patterns['where'], sql):
            user_prompt = match.group(1)
            system = match.group(2) if match.group(2) else 'lotus'
            operations['where'].append((user_prompt, system))
        
        # Extract semantic group by
        operations['group'] = []
        for match in re.finditer(self.patterns['group'], sql):
            column = match.group(1).strip()
            number_of_groups = int(match.group(2))
            system = match.group(3) if match.group(3) else 'lotus'
            operations['group'].append((column, number_of_groups, system))
        
        # Extract semantic aggregations
        operations['agg'] = []
        for match in re.finditer(self.patterns['agg'], sql):
            column = match.group(1)  # Could be None for global aggregation
            user_prompt = match.group(2)
            system = match.group(3) if match.group(3) else 'lotus'
            alias = match.group(4)

            # Determine which variation this is:
            if column is None:
                # Variation 1: SEM_AGG('user_prompt') AS alias
                # Variation 2: SEM_AGG('user_prompt', 'system') AS alias
                operations['agg'].append((None, user_prompt, system, alias))
            else:
                # Variation 3: SEM_AGG(column, 'user_prompt', 'system') AS alias
                # If no system specified but column exists, default to docetl
                if system == 'lotus' and column:
                    system = 'docetl'
                    logging.info(f"Defaulting system to 'docetl' for column-based aggregation: {column}")
                operations['agg'].append((column.strip(), user_prompt, system, alias))
        
        # Extract semantic select
        operations['select'] = []
        for match in re.finditer(self.patterns['select'], sql):
            user_prompt = match.group(1)
            system = match.group(2) if match.group(2) else 'lotus'
            alias = match.group(3)
            operations['select'].append((user_prompt, system, alias))
        
        # Extract semantic distinct
        operations['distinct'] = []
        for match in re.finditer(self.patterns['distinct'], sql):
            column = match.group(1).strip()
            system = match.group(2) if match.group(2) else 'lotus'
            operations['distinct'].append((column, system))
        
        # Extract semantic order by
        operations['order'] = []
        for match in re.finditer(self.patterns['order'], sql):
            column = match.group(1)
            user_prompt = match.group(2)
            system = match.group(3) if match.group(3) else 'lotus'
        
            if column is None:
                # Variation 1: SEM_ORDER_BY('user_prompt')
                # Variation 2: SEM_ORDER_BY('user_prompt', 'system')
                operations['order'].append((None, user_prompt, system))
            else:
                if system == 'lotus' and column:
                    system = 'docetl'
                    logging.info(f"Defaulting system to 'docetl' for column-based ordering: {column}")
                # Variation 3: SEM_ORDER_BY(column, 'user_prompt', 'system')
                operations['order'].append((column.strip(), user_prompt, system))
        
        return operations
    
    def extract_intersect_except_operations(self, sql: str) -> Dict[str, List[Tuple]]:
        """Extract INTERSECT and EXCEPT operations."""
        operations = {
            'intersect_all': [], 'intersect_set': [],
            'except_all': [], 'except_set': []
        }

        # Extract INTERSECT_ALL
        for m in re.finditer(self.patterns['intersect_all'], sql, flags=re.DOTALL | re.IGNORECASE):
            q1, q2 = self.split_top_level_args(m.group(1))[:2]
            operations['intersect_all'].append((m.span(), q1, q2))
        
        # Extract INTERSECT (set version)
        for m in re.finditer(self.patterns['intersect'], sql, flags=re.DOTALL | re.IGNORECASE):
            q1, q2 = self.split_top_level_args(m.group(1))[:2]
            operations['intersect_set'].append((m.span(), q1, q2))
        
        # Extract EXCEPT_ALL
        for m in re.finditer(self.patterns['except_all'], sql, flags=re.DOTALL | re.IGNORECASE):
            q1, q2 = self.split_top_level_args(m.group(1))[:2]
            operations['except_all'].append((m.span(), q1, q2))

        # Extract EXCEPT (set version)
        for m in re.finditer(self.patterns['except'], sql, flags=re.DOTALL | re.IGNORECASE):
            q1, q2 = self.split_top_level_args(m.group(1))[:2]
            operations['except_set'].append((m.span(), q1, q2))

        return operations
    
    @staticmethod
    def split_top_level_args(arg_str: str) -> List[str]:
        """Split comma-separated arguments, ignoring commas inside parentheses or quotes."""
        depth, in_q, buf, parts = 0, False, [], []
        for ch in arg_str:
            if ch in "\"'" and (not buf or buf[-1] != "\\"):
                in_q = not in_q
            elif not in_q:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    parts.append("".join(buf).strip()); buf = []
                    continue
            buf.append(ch)
        parts.append("".join(buf).strip())
        return parts

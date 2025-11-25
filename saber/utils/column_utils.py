"""
Utility functions for column operations.
"""
import re
import pandas as pd
from typing import Dict

def quote_dot_columns(sql: str, df_columns: list[str], db_type: str = 'duckdb') -> str:
    """
    Automatically quote column names containing dots or special characters in SQL.
    
    Args:
        sql: SQL query string
        df_columns: List of column names from DataFrame
        db_type: Database type ('duckdb', 'sqlite', 'mysql')
        
    Returns:
        SQL with quoted column names
    """
    # Use appropriate quote character based on database type
    if db_type == 'mysql':
        quote_char = '`'
    else:
        quote_char = '"'
    
    # Sort by length descending to handle longer column names first
    # This prevents partial matches for columns like "name" and "full_name"
    sorted_cols = sorted(df_columns, key=len, reverse=True)
    
    for col in sorted_cols:
        # Check if column needs quoting (contains dots, spaces, or special chars)
        needs_quoting = ('.' in col or ' ' in col or 
                        any(c in col for c in ['°', '℃', '℉', '±', '×', '÷', '-', '/', '(', ')']))
        
        if needs_quoting:
            # Skip if already quoted
            if (col.startswith('"') and col.endswith('"')) or (col.startswith('`') and col.endswith('`')):
                continue
            
            # Quote the column name
            quoted_col = f'{quote_char}{col}{quote_char}'
            
            # Replace unquoted occurrences with quoted ones
            # Use word boundaries to avoid partial matches
            # But also handle column names in various contexts (SELECT, WHERE, ORDER BY, etc.)
            pattern = rf'\b{re.escape(col)}\b'
            sql = re.sub(pattern, quoted_col, sql)
    
    return sql

def get_column_context(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Create a string describing the table schemas for LLM context.
    
    Args:
        dataframes: Dictionary of table name to DataFrame
        
    Returns:
        Schema description string
    """
    context = "Schema Information:\n"
    for name, df in dataframes.items():
        context += f"Table '{name}':\n"
        for col in df.columns:
            context += f"  - {col}\n"
    return context

def extract_table_aliases(sql: str) -> Dict[str, str]:
    """
    Extract table aliases from SQL query.
    
    Args:
        sql: SQL query string
        
    Returns:
        Dictionary mapping table names to aliases
    """
    aliases = {}
    # A more robust regex to capture aliases from FROM and JOIN clauses
    from_join_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)(?:\s+AS)?\s+([a-zA-Z0-9_]+)'
    for table, alias in re.findall(from_join_pattern, sql, re.IGNORECASE):
        # if table != alias:  # Only add if there's actually an alias
        aliases[alias] = table
    return aliases

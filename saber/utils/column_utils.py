"""
Utility functions for column operations.
"""
import re
import pandas as pd
from typing import Dict

def quote_dot_columns(sql: str, df_columns: list[str]) -> str:
    """
    Automatically quote column names containing dots in SQL.
    
    Args:
        sql: SQL query string
        df_columns: List of column names from DataFrame
        
    Returns:
        SQL with quoted column names
    """
    for col in df_columns:
        if '.' in col and not col.startswith('"') and not col.endswith('"'):
            # Quote the column name if it contains dots
            quoted_col = f'"{col}"'
            # Replace unquoted occurrences with quoted ones
            sql = re.sub(rf'\b{re.escape(col)}\b', quoted_col, sql)
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

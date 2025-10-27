"""
Utility functions for DataFrame operations.
"""
import pandas as pd
from typing import Tuple, Dict

def prepare_df_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Prepare DataFrame by replacing dots in column names with underscores.
    Returns the modified DataFrame and a mapping for restoration.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (modified_df, column_mapping)
    """
    column_mapping = {}
    new_columns = []
    
    for col in df.columns:
        if '.' in col:
            new_col = col.replace('.', '_')
            column_mapping[new_col] = col  # Map new -> original
            new_columns.append(new_col)
        else:
            new_columns.append(col)
    
    df_prepared = df.copy()
    df_prepared.columns = new_columns
    
    return df_prepared, column_mapping

def restore_df_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Restore original column names after processing.
    
    Args:
        df: DataFrame with modified column names
        column_mapping: Mapping from modified names to original names
        
    Returns:
        DataFrame with restored column names
    """
    df_restored = df.copy()
    
    # Restore original column names
    new_columns = []
    for col in df_restored.columns:
        if col in column_mapping:
            new_columns.append(column_mapping[col])
        else:
            new_columns.append(col)
    
    df_restored.columns = new_columns
    return df_restored

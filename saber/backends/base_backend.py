"""
Base backend interface for SABER semantic operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd


class BaseBackend(ABC):
    """Abstract base class for SABER backends."""
    
    def __init__(self, name: str, api_key: str = None, model: str = None, api_base: str = None, embedding_model: str = None, embedding_api_base: str = None):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.embedding_model = embedding_model
        self.embedding_api_base = embedding_api_base
    
    def set_api_key(self, api_key: str):
        """Set or update the API key for this backend."""
        self.api_key = api_key

    def set_model_config(self, model: str, api_base: str = None, api_key: str = None, embedding_model: str = None, embedding_api_base: str = None):
        """Set or update the model configuration for this backend."""
        self.model = model
        self.api_base = api_base
        if api_key is not None:
            self.api_key = api_key
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if embedding_api_base is not None:
            self.embedding_api_base = embedding_api_base
    
    def _ensure_unique_alias(self, df: pd.DataFrame, alias: str) -> str:
        """
        Ensure the alias doesn't conflict with existing column names (case-insensitive).
        
        Args:
            df: DataFrame with existing columns
            alias: Proposed alias name
            
        Returns:
            Unique alias name
        """
        # Get existing column names in lowercase for case-insensitive comparison
        existing_cols_lower = {col.lower() for col in df.columns}
        
        # Check if alias conflicts
        if alias.lower() not in existing_cols_lower:
            return alias
        
        # Find a unique name by appending a suffix
        suffix = 1
        while f"{alias}_{suffix}".lower() in existing_cols_lower:
            suffix += 1
        
        return f"{alias}_{suffix}"
    
    @abstractmethod
    def prepare_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
        """
        Prepare DataFrame for backend processing.
        Returns the modified DataFrame and a column mapping for restoration.
        """
        pass
    
    @abstractmethod
    def restore_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Restore original column names after backend processing."""
        pass
    
    @abstractmethod
    def update_prompt(self, prompt: str, column_mapping: Dict[str, str]) -> str:
        """Update prompt for backend-specific column references."""
        pass
    
    @abstractmethod
    def sem_where(self, df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
        """Semantic filtering operation."""
        pass
    
    @abstractmethod
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection/mapping operation."""
        pass
    
    @abstractmethod
    def sem_join(self, df1: pd.DataFrame, df2: pd.DataFrame, user_prompt: str, 
                df1_name: str = "left", df2_name: str = "right") -> pd.DataFrame:
        """Semantic join operation."""
        pass
    
    @abstractmethod
    def sem_group_by(self, df: pd.DataFrame, column: str, number_of_groups: int) -> pd.DataFrame:
        """Semantic grouping operation."""
        pass
    
    @abstractmethod
    def sem_agg(self, df: pd.DataFrame, user_prompt: str, alias: str, 
               group_by_col: Optional[List[str]] = None, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic aggregation operation."""
        pass
    
    @abstractmethod
    def sem_distinct(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Semantic deduplication operation."""
        pass
    
    @abstractmethod
    def sem_order_by(self, df: pd.DataFrame, user_prompt: str, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic ordering operation."""
        pass
    
    def supports_operation(self, operation: str) -> bool:
        """Check if backend supports a specific operation."""
        supported_ops = [
            'sem_filter', 'sem_select', 'sem_join', 'sem_group_by', 
            'sem_agg', 'sem_distinct', 'sem_order_by'
        ]
        return operation in supported_ops

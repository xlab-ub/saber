"""
LOTUS backend implementation for SABER semantic operations.
"""
import re
import time
from typing import Dict, Optional, List
import logging
import pandas as pd

from .base_backend import BaseBackend
from ..config import LOTUS_DEFAULT_DEDUP_THRESHOLD
from ..benchmark import BenchmarkStats, extract_lotus_stats

logger = logging.getLogger(__name__)

class LOTUSBackend(BaseBackend):
    """LOTUS backend for semantic operations using embedding-based similarity."""
    
    def __init__(self, api_key: str = None, model: str = None, api_base: str = None):
        super().__init__("LOTUS", api_key, model, api_base)
        self.last_stats = BenchmarkStats()
        self._lm = None  # Will be set by engine
    
    def _track_operation(self, operation_func, *args, **kwargs):
        """Track cost and latency for a LOTUS operation."""
        if self._lm is None:
            # If LM is not set, just run the operation without tracking
            return operation_func(*args, **kwargs)
        
        # Reset LM stats before operation
        self._lm.reset_stats()
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # Extract stats from LM
            self.last_stats = extract_lotus_stats(self._lm)
            self.last_stats.total_semantic_execution_time_seconds = elapsed_time
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.last_stats = BenchmarkStats(
                total_semantic_execution_time_seconds=elapsed_time
            )
            raise
    
    def prepare_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
        """Replace dots in column names with underscores for LOTUS."""
        column_mapping = {}
        new_columns = []
        
        for col in df.columns:
            if '.' in col:
                new_col = col.replace('.', '_')
                column_mapping[new_col] = col
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        
        df_lotus = df.copy()
        df_lotus.columns = new_columns
        
        return df_lotus, column_mapping
    
    def restore_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Restore original column names after LOTUS processing."""
        df_restored = df.copy()
        
        new_columns = []
        for col in df_restored.columns:
            original_col = column_mapping.get(col, col)
            new_columns.append(original_col)
        
        df_restored.columns = new_columns
        return df_restored
    
    def update_prompt(self, prompt: str, column_mapping: Dict[str, str]) -> str:
        """Update prompts to use underscore column names for LOTUS."""
        updated_prompt = prompt
        
        # Replace dot notation with underscore notation in prompts
        for underscore_col, original_col in column_mapping.items():
            # Replace {original.col} with {underscore_col}
            pattern = r'\{' + re.escape(original_col) + r'\}'
            replacement = '{' + underscore_col + '}'
            updated_prompt = re.sub(pattern, replacement, updated_prompt)
        
        return updated_prompt
    
    def sem_where(self, df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
        """Semantic filtering using LOTUS."""
        def _operation():
            df_lotus, column_mapping = self.prepare_dataframe(df)
            updated_prompt = self.update_prompt(user_prompt, column_mapping)
            
            result_df_lotus = df_lotus.sem_filter(updated_prompt)
            result_df = self.restore_dataframe(result_df_lotus, column_mapping)
            
            logging.info(f"Result of LOTUS SEM_WHERE: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_WHERE: {e}")
            return df
    
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection using LOTUS."""
        def _operation():
            df_lotus, column_mapping = self.prepare_dataframe(df)
            updated_prompt = self.update_prompt(user_prompt, column_mapping)
            
            result_df_lotus = df_lotus.sem_map(updated_prompt, suffix=alias)
            result_df = self.restore_dataframe(result_df_lotus, column_mapping)

            logging.info(f"Result of LOTUS SEM_SELECT: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_SELECT: {e}")
            raise  # Re-raise the exception instead of returning original df
    
    def sem_join(self, df1: pd.DataFrame, df2: pd.DataFrame, user_prompt: str,
                df1_name: str = "left", df2_name: str = "right") -> pd.DataFrame:
        """Semantic join using LOTUS."""
        def rename_sem_join_columns(df, left_table, right_table, separator=':'):
            new_columns = []
            for col in df.columns:
                # Replace separator+left and separator+right suffixes
                col = re.sub(rf'^(.*)({re.escape(separator)})left$', rf'{left_table}.\1', col)
                col = re.sub(rf'^(.*)({re.escape(separator)})right$', rf'{right_table}.\1', col)
                new_columns.append(col)
            df.columns = new_columns
            return df
        
        def _operation():
            result_df_lotus = df1.sem_join(df2, user_prompt)
            result_df = rename_sem_join_columns(result_df_lotus, df1_name, df2_name)
            return result_df
        
        try: 
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_JOIN: {e}")
            return pd.DataFrame()
    
    def sem_group_by(self, df: pd.DataFrame, column: str, number_of_groups: int) -> pd.DataFrame:
        """Semantic grouping using LOTUS."""
        logging.info(f"Grouping by {column} into {number_of_groups} groups using LOTUS.")
        logging.info(f"Original DataFrame:\n{df.head(10)}")
        
        def _operation():
            result_df = df.sem_index(column, f"{column.replace(' ', '_')}_index").sem_cluster_by(column, number_of_groups)
            logging.info(f"Result of LOTUS SEM_GROUP_BY: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_GROUP_BY: {e}")
            return df
    
    def sem_agg(self, df: pd.DataFrame, user_prompt: str, alias: str,
               group_by_col: Optional[List[str]] = None, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic aggregation using LOTUS."""
        def _operation():
            if group_by_col:
                result_df = df.sem_agg(user_prompt, suffix=alias, group_by=group_by_col)
            else:
                result_df = df.sem_agg(user_prompt, suffix=alias)
            logging.info(f"Result of LOTUS SEM_AGG: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_AGG: {e}")
            return df
    
    def sem_distinct(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Semantic deduplication using LOTUS."""
        def _operation():
            result_df = df.sem_index(f"{column}", f"{column}_index").sem_dedup(f"{column}", threshold=LOTUS_DEFAULT_DEDUP_THRESHOLD)
            logging.info(f"Result of LOTUS SEM_DISTINCT: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_DISTINCT: {e}")
            return df
    
    def sem_order_by(self, df: pd.DataFrame, user_prompt: str, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic ordering using LOTUS."""
        def _operation():
            number_of_records = df.shape[0]
            if number_of_records == 0:
                logging.info("DataFrame is empty, returning original DataFrame.")
                return df
            result_df = df.sem_topk(user_prompt, K=number_of_records, return_stats=False)
            logging.info(f"Result of LOTUS SEM_ORDER_BY: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_ORDER_BY: {e}")
            return df

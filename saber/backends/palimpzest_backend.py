"""
Palimpzest backend implementation for SABER semantic operations.
"""
import re
import os
import time
import threading
from contextlib import contextmanager
from typing import Dict, Optional, List
import logging
import pandas as pd
import traceback
import litellm

import palimpzest as pz
from palimpzest.core.elements.groupbysig import GroupBySig
import chromadb
# from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

from .base_backend import BaseBackend
# from ..config import PALIMPZEST_DEFAULT_MODEL, PALIMPZEST_DEFAULT_EMBEDDING_MODEL, LOCAL_EMBEDDING_API_BASE
from ..benchmark import BenchmarkStats, extract_palimpzest_stats

logger = logging.getLogger(__name__)

# TODO: Add support for un-registered vllm models in Palimpzest

@contextmanager
def temporary_env_var(var_name: str, value: str):
    """Temporarily set an environment variable and restore it afterwards."""
    original_value = os.environ.get(var_name)
    
    if value is not None:
        os.environ[var_name] = value
    
    try:
        yield
    finally:
        if original_value is not None:
            os.environ[var_name] = original_value
        elif var_name in os.environ:
            del os.environ[var_name]

@contextmanager
def temporary_env_vars(env_vars: Dict[str, str]):
    """Temporarily set multiple environment variables and restore them afterwards."""
    original_values = {}
    for var_name, value in env_vars.items():
        original_values[var_name] = os.environ.get(var_name)
        if value is not None:
            os.environ[var_name] = value

    try:
        yield
    finally:
        for var_name, original_value in original_values.items():
            if original_value is not None:
                os.environ[var_name] = original_value
            elif var_name in os.environ:
                del os.environ[var_name]


class PalimpzestBackend(BaseBackend):
    """Palimpzest backend for semantic operations using symbolic programming."""
    
    def __init__(self, api_key: str = None, model: str = None, api_base: str = None, embedding_model: str = None, embedding_api_base: str = None):
        super().__init__("palimpzest", api_key, model, api_base, embedding_model, embedding_api_base)
        # chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.chroma_client = chromadb.Client()
        self.last_stats = BenchmarkStats()
        if not hasattr(pz, 'MemoryDataset'):
            self._patch_palimpzest_to_df()
        # Set default model if not provided
        if self.model is None:
            from ..config import PALIMPZEST_DEFAULT_MODEL
            self.model = PALIMPZEST_DEFAULT_MODEL
        if self.embedding_model is None:
            from ..config import PALIMPZEST_DEFAULT_EMBEDDING_MODEL
            self.embedding_model = PALIMPZEST_DEFAULT_EMBEDDING_MODEL
        self._api_call_count = 0
        self._embedding_call_count = 0
        self._api_call_lock = threading.Lock()
        self._register_callbacks()
    
    def _register_callbacks(self):
        """Register litellm callbacks for API call counting."""
        # Initialize callback lists if they don't exist or are not lists
        if not hasattr(litellm, 'success_callback'):
            litellm.success_callback = []
        elif litellm.success_callback is None:
            litellm.success_callback = []
        elif not isinstance(litellm.success_callback, list):
            litellm.success_callback = [litellm.success_callback]
            
        if not hasattr(litellm, 'failure_callback'):
            litellm.failure_callback = []
        elif litellm.failure_callback is None:
            litellm.failure_callback = []
        elif not isinstance(litellm.failure_callback, list):
            litellm.failure_callback = [litellm.failure_callback]
        
        # Append callbacks if not already registered (idempotent)
        if self._success_callback not in litellm.success_callback:
            litellm.success_callback.append(self._success_callback)
        if self._failure_callback not in litellm.failure_callback:
            litellm.failure_callback.append(self._failure_callback)
    
    def _success_callback(self, kwargs, completion_response, start_time, end_time):
        """Callback for successful API calls."""
        with self._api_call_lock:
            call_type = kwargs.get('call_type', '')
            if call_type in ('embedding', 'aembedding'):
                self._embedding_call_count += 1
            else:
                self._api_call_count += 1
    
    def _failure_callback(self, kwargs, completion_response, start_time, end_time):
        """Callback for failed API calls."""
        with self._api_call_lock:
            call_type = kwargs.get('call_type', '')
            if call_type in ('embedding', 'aembedding'):
                self._embedding_call_count += 1
            else:
                self._api_call_count += 1
    
    def _reset_api_call_count(self):
        """Reset API call counter."""
        with self._api_call_lock:
            self._api_call_count = 0
            self._embedding_call_count = 0
    
    def _get_api_call_count(self) -> int:
        """Get current API call count.
        
        Includes a small delay to ensure litellm async callbacks complete
        before reading the counter.
        """
        import time
        time.sleep(0.01)  # 10ms to flush callback queue
        with self._api_call_lock:
            return self._api_call_count
    
    def _get_embedding_call_count(self) -> int:
        """Get current embedding call count.
        
        Includes a small delay to ensure litellm async callbacks complete
        before reading the counter.
        """
        import time
        time.sleep(0.01)  # 10ms to flush callback queue
        with self._api_call_lock:
            return self._embedding_call_count
    
    def _wrap_embedding_function(self, ef):
        """Wrap ChromaDB embedding function to track calls."""
        if ef is None or hasattr(ef, '_tracking_wrapped'):
            return ef
        
        if hasattr(ef, '__call__'):
            original_call = ef.__call__
            backend = self
            
            def tracked_call(input):
                with backend._api_call_lock:
                    backend._embedding_call_count += 1
                return original_call(input)
            
            ef.__call__ = tracked_call
            ef._tracking_wrapped = True
        
        return ef
    
    def __del__(self):
        """Cleanup when destroyed."""
        if not hasattr(pz, 'MemoryDataset'):
            self._restore_original_to_df()
    
    def prepare_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
        """Replace dots in column names with underscores for Palimpzest."""
        column_mapping = {}
        new_columns = []
        
        for col in df.columns:
            if '.' in col:
                new_col = col.replace('.', '_')
                column_mapping[new_col] = col
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        
        df_palimpzest = df.copy()
        df_palimpzest.columns = new_columns
        
        return df_palimpzest, column_mapping
    
    def restore_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Restore original column names after Palimpzest processing."""
        df_restored = df.copy()
        
        new_columns = []
        for col in df_restored.columns:
            original_col = column_mapping.get(col, col)
            new_columns.append(original_col)
        
        df_restored.columns = new_columns
        return df_restored
    
    def update_prompt(self, prompt: str, column_mapping: Dict[str, str]) -> str:
        """Update prompts - Palimpzest uses original prompts without modification."""
        return prompt  # Palimpzest doesn't require prompt modification
    
    def _patch_palimpzest_to_df(self):
        """Patch Palimpzest's DataRecord methods to handle mixed types safely"""
        try:
            from palimpzest.core.elements.records import DataRecord
            
            # Store the original methods
            self._original_to_df = DataRecord.to_df
            self._original_init = DataRecord.__init__
            
            # Patch the __init__ method to handle type conversion
            def safe_init(self, schema, source_idx, parent_id=None, cardinality_idx=None):
                # Ensure source_idx is provided
                assert source_idx is not None, "Every DataRecord must be constructed with a source_idx"

                # schema for the data record
                self.schema = schema

                # mapping from field names to Field objects
                self.field_types = schema.field_map()

                # mapping from field names to their values
                self.field_values = {}

                # the index in the DataReader from which this DataRecord is derived
                self.source_idx = int(source_idx)

                # Convert parent_id to string if it's not None
                self.parent_id = str(parent_id) if parent_id is not None else parent_id

                # store the cardinality index
                self.cardinality_idx = cardinality_idx

                # indicator variable for filter operations
                self.passed_operator = True

                # Generate ID string with proper type conversion
                if cardinality_idx is None:
                    id_str = str(schema) + (self.parent_id if self.parent_id is not None else str(self.source_idx))
                else:
                    id_str = str(schema) + str(cardinality_idx) + (self.parent_id if self.parent_id is not None else str(self.source_idx))
                
                # Import hash function
                from palimpzest.utils.hash_helpers import hash_for_id
                self.id = hash_for_id(id_str)
            
            @staticmethod
            def safe_to_df(records, project_cols=None):
                if len(records) == 0:
                    return pd.DataFrame()

                try:
                    fields = records[0].get_field_names()
                    if project_cols is not None and len(project_cols) > 0:
                        fields = [field for field in fields if field in project_cols]

                    # Safely extract values with type preservation
                    data = []
                    for record in records:
                        row_data = {}
                        for k in fields:
                            try:
                                # Try multiple ways to access the field value safely
                                if hasattr(record, 'field_values') and k in record.field_values:
                                    value = record.field_values[k]
                                else:
                                    # Fallback to direct attribute access
                                    value = getattr(record, k, "")
                                
                                # Handle different types appropriately
                                if value is None:
                                    value = None  # Keep None as None
                                elif isinstance(value, (int, float)):
                                    # Keep numeric types as-is to preserve data types
                                    value = value
                                elif isinstance(value, bool):
                                    # Keep boolean as-is
                                    value = value
                                elif isinstance(value, str):
                                    # Keep strings as-is
                                    value = value
                                elif isinstance(value, (list, dict, tuple)):
                                    # Convert complex types to strings
                                    value = str(value)
                                else:
                                    # For other unknown types, convert to string
                                    value = str(value)
                                
                                row_data[k] = value
                            except Exception as e:
                                logging.warning(f"Error accessing field '{k}': {e}")
                                row_data[k] = None  # Use None instead of empty string
                        data.append(row_data)
                    
                    df = pd.DataFrame(data)
                    
                    # Try to infer and restore appropriate data types
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Try to convert to numeric if possible
                            try:
                                # First check if all non-null values are numeric
                                non_null_values = df[col].dropna()
                                if len(non_null_values) > 0:
                                    # Try converting to numeric
                                    numeric_series = pd.to_numeric(non_null_values, errors='coerce')
                                    if not numeric_series.isna().all():
                                        # If conversion was successful for at least some values
                                        df[col] = pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass  # Keep as object type if conversion fails
                    
                    return df
                    
                except Exception as e:
                    logging.error(f"Error in safe_to_df: {e}")
                    # Last resort: return empty DataFrame with error info
                    return pd.DataFrame([{"error": str(e)}])
            
            # Replace the methods
            DataRecord.__init__ = safe_init
            DataRecord.to_df = safe_to_df
            logging.info("Successfully patched DataRecord.__init__ and to_df methods")
            
        except ImportError:
            logging.warning("Could not import DataRecord, skipping patch")
        except Exception as e:
            logging.warning(f"Could not patch DataRecord methods: {e}")

    def _restore_original_to_df(self):
        """Restore the original methods"""
        try:
            from palimpzest.core.elements.records import DataRecord
            if hasattr(self, '_original_to_df') and self._original_to_df is not None:
                DataRecord.to_df = self._original_to_df
            if hasattr(self, '_original_init') and self._original_init is not None:
                DataRecord.__init__ = self._original_init
            logging.info("Restored original DataRecord methods")
        except Exception as e:
            logging.warning(f"Could not restore original methods: {e}")

    def _sem_filter_with_limit(self, df: pd.DataFrame, user_prompt: str, column_mapping: Dict, limit: int) -> pd.DataFrame:
        """Execute semantic filter with early termination via iterative batching.
        
        This optimization processes the DataFrame in batches and stops once
        enough matching rows are found. This can significantly reduce the
        number of LLM calls for queries with small LIMIT values.
        
        Args:
            df: Input DataFrame (already prepared for Palimpzest)
            user_prompt: The semantic filter prompt
            column_mapping: Column mapping for restoring original names
            limit: Target number of matching rows to find
            
        Returns:
            DataFrame with at least `limit` matching rows (if available),
            or all matching rows if fewer than `limit` exist.
        """
        total_rows = len(df)
        
        # If DataFrame is small enough, just process all at once
        # Threshold: 2x the limit or 100 rows, whichever is smaller
        small_df_threshold = min(limit * 2, 100)
        if total_rows <= small_df_threshold:
            logger.debug(f"DataFrame small ({total_rows} rows), processing all at once")
            return self._run_sem_filter(df, user_prompt)
        
        # For larger DataFrames, use iterative batch processing
        # Batch size: Start with 2x the limit to account for selectivity,
        # but ensure minimum of 50 rows per batch for efficiency
        batch_size = max(limit * 2, 50)
        
        collected_results = []
        collected_count = 0
        processed_rows = 0
        
        logger.info(f"LIMIT pushdown enabled: targeting {limit} rows from {total_rows} total")
        
        while processed_rows < total_rows and collected_count < limit:
            # Get the next batch
            batch_end = min(processed_rows + batch_size, total_rows)
            batch_df = df.iloc[processed_rows:batch_end].copy()
            
            logger.debug(f"Processing batch [{processed_rows}:{batch_end}] ({len(batch_df)} rows)")
            
            # Apply semantic filter to this batch
            batch_result = self._run_sem_filter(batch_df, user_prompt)
            
            # Collect results
            if not batch_result.empty:
                collected_results.append(batch_result)
                collected_count += len(batch_result)
                logger.debug(f"Batch yielded {len(batch_result)} matches, total: {collected_count}/{limit}")
            
            processed_rows = batch_end
            
            # Early termination: we have enough results
            if collected_count >= limit:
                logger.info(f"Early termination: found {collected_count} matches after processing "
                           f"{processed_rows}/{total_rows} rows ({100*processed_rows/total_rows:.1f}%)")
                break
        
        # Combine all collected results
        if collected_results:
            result = pd.concat(collected_results, ignore_index=True)
            # Trim to exactly the limit if we got more
            if len(result) > limit:
                result = result.head(limit)
            return result
        else:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=df.columns)

    def _run_sem_filter(self, df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
        """Run Palimpzest semantic filter on a DataFrame.
        
        This is a helper method that encapsulates the core Palimpzest filter logic,
        used both for regular filtering and batch processing with limit hints.
        
        Args:
            df: Input DataFrame (already prepared for Palimpzest)
            user_prompt: The semantic filter prompt
            
        Returns:
            Filtered DataFrame
        """
        with temporary_env_var("OPENAI_API_KEY", self.api_key):
            if hasattr(pz, 'MemoryDataset'):
                pz_dataset = pz.MemoryDataset(id="temp_dataset", vals=df)
            else:
                pz_dataset = pz.Dataset(source=df)
            pz_dataset = pz_dataset.sem_filter(user_prompt)
            output = pz_dataset.run(config=pz.QueryProcessorConfig(
                # default policy is MaxQuality
                available_models=[self.model],
                api_base=self.api_base,
            ))
            result_df = output.to_df()
            
            # Convert complex types (list/dict) to string to avoid MySQL type conversion errors
            for col in result_df.columns:
                if result_df[col].dtype == 'object':
                    result_df[col] = result_df[col].apply(
                        lambda x: str(x) if isinstance(x, (list, dict)) else x
                    )
            
            # Extract stats
            if hasattr(output, 'execution_stats'):
                self.last_stats = extract_palimpzest_stats(output.execution_stats)
            
            return result_df

    def sem_where(self, df: pd.DataFrame, user_prompt: str, limit_hint: int = None) -> pd.DataFrame:
        """Semantic filtering using Palimpzest.
        
        Args:
            df: Input DataFrame to filter
            user_prompt: The semantic filter prompt with column placeholders
            limit_hint: Optional hint for early termination. If provided, the filter
                       will stop processing once enough matching rows are found.
                       This is an optimization hint pushed down from LIMIT clauses.
        """
        try:
            self._reset_api_call_count()
            df_palimpzest, column_mapping = self.prepare_dataframe(df)

            # If limit_hint is provided, use iterative batch processing for early termination
            if limit_hint is not None and limit_hint > 0:
                result_df = self._sem_filter_with_limit(df_palimpzest, user_prompt, column_mapping, limit_hint)
            else:
                result_df = self._run_sem_filter(df_palimpzest, user_prompt)

            if column_mapping:
                result_df = self.restore_dataframe(result_df, column_mapping)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns)

            self.last_stats.api_calls = self._get_api_call_count()
            self.last_stats.embedding_calls = self._get_embedding_call_count()
            logging.info(f"Result of Palimpzest SEM_WHERE: {result_df.head(10)}")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_WHERE: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def sem_where_marker(self, df: pd.DataFrame, user_prompt: str, result_column: str = "_sem_where_result") -> pd.DataFrame:
        """Semantic filter marker using Palimpzest - adds boolean column instead of filtering.
        
        Unlike sem_where which filters rows, this method preserves all rows and adds
        a boolean column indicating which rows match the semantic filter condition.
        This is used for SEM_WHERE in SELECT clause (e.g., CASE WHEN SEM_WHERE(...)).
        
        Args:
            df: Input DataFrame
            user_prompt: The semantic filter prompt with column placeholders
            result_column: Name of the column to add with boolean results
        
        Returns:
            DataFrame with all original rows plus a boolean result column
        """
        try:
            self._reset_api_call_count()
            df_palimpzest, column_mapping = self.prepare_dataframe(df)

            # Use sem_filter to get the matching rows
            filtered_df = self._run_sem_filter(df_palimpzest, user_prompt)
            
            # Restore column names for comparison
            if column_mapping:
                filtered_df = self.restore_dataframe(filtered_df, column_mapping)
            
            # Get the indices of matching rows
            matching_indices = set(filtered_df.index)
            
            # Create result column: True for matching rows, False otherwise
            result_df = df.copy()
            result_df[result_column] = result_df.index.isin(matching_indices)
            
            self.last_stats.api_calls = self._get_api_call_count()
            self.last_stats.embedding_calls = self._get_embedding_call_count()
            logging.info(f"Result of Palimpzest SEM_WHERE_MARKER: {result_df[result_column].sum()} True out of {len(result_df)} rows")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_WHERE_MARKER: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Palimpzest SEM_WHERE_MARKER execution failed: {str(e)}") from e
    
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection using Palimpzest."""
        # Ensure alias doesn't conflict with existing columns
        unique_alias = self._ensure_unique_alias(df, alias)
        try:
            self._reset_api_call_count()
            df_palimpzest, column_mapping = self.prepare_dataframe(df)

            # Use temporary environment variable for API key
            with temporary_env_var("OPENAI_API_KEY", self.api_key):
                if hasattr(pz, 'MemoryDataset'):
                    pz_dataset = pz.MemoryDataset(id="temp_dataset", vals=df_palimpzest)
                    pz_dataset = pz_dataset.sem_map([{"name": unique_alias, "type": str, "desc": user_prompt}])
                else:
                    pz_dataset = pz.Dataset(source=df_palimpzest)
                    pz_dataset = pz_dataset.sem_add_columns([{"name": unique_alias, "type": str, "desc": user_prompt}])
                output = pz_dataset.run(config=pz.QueryProcessorConfig(
                    # default policy is MaxQuality
                    available_models=[self.model],
                    api_base=self.api_base,
                ))
                result_df = output.to_df()
                
                # Convert complex types (list/dict) to string to avoid MySQL type conversion errors
                for col in result_df.columns:
                    if result_df[col].dtype == 'object':
                        result_df[col] = result_df[col].apply(
                            lambda x: str(x) if isinstance(x, (list, dict)) else x
                        )
                
                # Extract stats
                if hasattr(output, 'execution_stats'):
                    self.last_stats = extract_palimpzest_stats(output.execution_stats)

            if not result_df.empty and column_mapping:
                result_df = self.restore_dataframe(result_df, column_mapping)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns.tolist() + [unique_alias])

            self.last_stats.api_calls = self._get_api_call_count()
            logging.info(f"Result of Palimpzest SEM_SELECT: {result_df.head(10)}")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_SELECT: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def sem_join(self, df1: pd.DataFrame, df2: pd.DataFrame, user_prompt: str,
                df1_name: str = "left", df2_name: str = "right",
                limit_hint: Optional[int] = None) -> pd.DataFrame:
        """Semantic join using Palimpzest.
        
        Args:
            df1: Left DataFrame
            df2: Right DataFrame  
            user_prompt: The semantic join condition prompt
            df1_name: Name/alias for the left table
            df2_name: Name/alias for the right table
            limit_hint: If set, use batched processing to limit LLM calls.
                       Will attempt to find at least limit_hint matches
                       using a multiplicative sampling approach.
        """
        def rename_sem_join_columns(df, left_table, right_table, separator='_'):
            new_columns = []
            for col in df.columns:
                m_right = re.match(rf'^(.*?){re.escape(separator)}right$', col)
                if m_right:
                    base = m_right.group(1)
                    new_col = f"{right_table}.{base}"
                else:
                    new_col = f"{left_table}.{col}"
                new_columns.append(new_col)
            df.columns = new_columns
            return df
        
        def do_join(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
            """Execute a single join operation."""
            left_df_pz, left_col_mapping = self.prepare_dataframe(left_df)
            right_df_pz, right_col_mapping = self.prepare_dataframe(right_df)
            col_mapping = {**left_col_mapping, **right_col_mapping}
            
            with temporary_env_var("OPENAI_API_KEY", self.api_key):
                left_pz_dataset = pz.MemoryDataset(id="temp_left_dataset", vals=left_df_pz)
                right_pz_dataset = pz.MemoryDataset(id="temp_right_dataset", vals=right_df_pz)
                joined_pz_dataset = left_pz_dataset.sem_join(right_pz_dataset, user_prompt)
                output = joined_pz_dataset.run(config=pz.QueryProcessorConfig(
                    available_models=[self.model],
                    api_base=self.api_base,
                ))
                result_df = output.to_df()
            
            result_df = rename_sem_join_columns(result_df, df1_name, df2_name)
            if not result_df.empty and col_mapping:
                result_df = self.restore_dataframe(result_df, col_mapping)
            return result_df
        
        if not hasattr(pz, 'MemoryDataset'):
            raise NotImplementedError("Palimpzest does not support semantic join operation directly.")
        
        try:
            self._reset_api_call_count()
            
            # If no limit hint or small tables, do full join
            if limit_hint is None or (len(df1) * len(df2) <= 1000):
                df1_palimpzest, df1_column_mapping = self.prepare_dataframe(df1)
                df2_palimpzest, df2_column_mapping = self.prepare_dataframe(df2)
                column_mapping = {**df1_column_mapping, **df2_column_mapping}

                # Use temporary environment variable for API key
                with temporary_env_var("OPENAI_API_KEY", self.api_key):
                    left_pz_dataset = pz.MemoryDataset(id="temp_left_dataset", vals=df1_palimpzest)
                    right_pz_dataset = pz.MemoryDataset(id="temp_right_dataset", vals=df2_palimpzest)
                    joined_pz_dataset = left_pz_dataset.sem_join(right_pz_dataset, user_prompt)
                    output = joined_pz_dataset.run(config=pz.QueryProcessorConfig(
                        # default policy is MaxQuality
                        available_models=[self.model],
                        api_base=self.api_base,
                    ))
                    result_df = output.to_df()
                
                # Extract stats
                if hasattr(output, 'execution_stats'):
                    self.last_stats = extract_palimpzest_stats(output.execution_stats)
                result_df = rename_sem_join_columns(result_df, df1_name, df2_name)

                if not result_df.empty and column_mapping:
                    result_df = self.restore_dataframe(result_df, column_mapping)

                self.last_stats.api_calls = self._get_api_call_count()
                self.last_stats.embedding_calls = self._get_embedding_call_count()
                logging.info(f"Result of Palimpzest SEM_JOIN: {result_df.head(10)}")
                return result_df
            
            # Batched approach for limit optimization
            # Strategy: Start with a sample, expand if needed
            results = []
            total_matches = 0
            
            # Initial batch sizes - use multiplicative factor of the limit
            # Since not all pairs match, we need more rows to find enough matches
            expansion_factor = 3  # Start with 3x to have buffer for non-matches
            initial_left_size = min(len(df1), max(10, limit_hint * expansion_factor))
            initial_right_size = min(len(df2), max(10, limit_hint * expansion_factor))
            
            left_processed = 0
            right_processed = 0
            
            # First batch - sample from both tables
            left_batch = df1.iloc[:initial_left_size]
            right_batch = df2.iloc[:initial_right_size]
            
            logger.info(f"SEM_JOIN with limit_hint={limit_hint}: Starting with batch sizes "
                       f"left={len(left_batch)}/{len(df1)}, right={len(right_batch)}/{len(df2)}")
            
            result_df = do_join(left_batch, right_batch)
            total_matches = len(result_df)
            
            if total_matches > 0:
                results.append(result_df)
            
            left_processed = len(left_batch)
            right_processed = len(right_batch)
            
            # If we have enough matches, return early
            if total_matches >= limit_hint:
                logger.info(f"SEM_JOIN early termination: Found {total_matches} matches "
                           f"with {left_processed}x{right_processed} = {left_processed * right_processed} LLM calls "
                           f"(saved {len(df1) * len(df2) - left_processed * right_processed} calls)")
                combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                self.last_stats.api_calls = self._get_api_call_count()
                self.last_stats.embedding_calls = self._get_embedding_call_count()
                return combined.head(limit_hint) if len(combined) > limit_hint else combined
            
            # Expand batches iteratively if needed
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while total_matches < limit_hint and iteration < max_iterations:
                iteration += 1
                
                # Expand: double the batch sizes
                new_left_end = min(len(df1), left_processed * 2)
                new_right_end = min(len(df2), right_processed * 2)
                
                # If we can't expand anymore, break
                if new_left_end == left_processed and new_right_end == right_processed:
                    logger.info(f"SEM_JOIN: All rows processed, found {total_matches} matches")
                    break
                
                # Process new left rows with all processed right rows
                if new_left_end > left_processed:
                    left_new = df1.iloc[left_processed:new_left_end]
                    right_old = df2.iloc[:right_processed]
                    
                    if len(left_new) > 0 and len(right_old) > 0:
                        result_df = do_join(left_new, right_old)
                        if len(result_df) > 0:
                            results.append(result_df)
                            total_matches += len(result_df)
                
                # Process new right rows with all left rows (including new ones)
                if new_right_end > right_processed:
                    left_all = df1.iloc[:new_left_end]
                    right_new = df2.iloc[right_processed:new_right_end]
                    
                    if len(left_all) > 0 and len(right_new) > 0:
                        result_df = do_join(left_all, right_new)
                        if len(result_df) > 0:
                            results.append(result_df)
                            total_matches += len(result_df)
                
                left_processed = new_left_end
                right_processed = new_right_end
                
                logger.info(f"SEM_JOIN iteration {iteration}: {total_matches} matches found, "
                           f"processed {left_processed}x{right_processed}")
                
                if total_matches >= limit_hint:
                    break
            
            logger.info(f"SEM_JOIN completed: Found {total_matches} matches "
                       f"with {left_processed}x{right_processed} rows processed "
                       f"(full join would be {len(df1)}x{len(df2)})")
            
            combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
            # Remove potential duplicates from overlapping batches
            if not combined.empty:
                combined = combined.drop_duplicates()
            
            self.last_stats.api_calls = self._get_api_call_count()
            self.last_stats.embedding_calls = self._get_embedding_call_count()
            logging.info(f"Result of Palimpzest SEM_JOIN: {combined.head(10)}")
            return combined.head(limit_hint) if limit_hint and len(combined) > limit_hint else combined
            
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_JOIN: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def sem_group_by(self, df: pd.DataFrame, column: str, number_of_groups: int) -> pd.DataFrame:
        """Semantic grouping using Palimpzest - temporarily disabled due to upstream bug."""
        # TODO: Temporarily disabled due to Palimpzest bug
        # Issue: https://github.com/mitdbg/palimpzest/issues/185
        # logging.warning("Palimpzest groupby is temporarily disabled due to upstream bug")
        # raise NotImplementedError("Palimpzest groupby is temporarily disabled due to upstream bug (issue #185)")
        # result_df = self._run_palimpzest_groupby(df, column, number_of_groups)
        raise NotImplementedError("Palimpzest does not fully support semantic group by operation directly.")
    
    def sem_agg(self, df: pd.DataFrame, user_prompt: str, alias: str,
               group_by_col: Optional[List[str]] = None, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic aggregation using Palimpzest."""
        # TODO: Temporarily disabled due to error "... validation errors for Schema..."
        raise NotImplementedError("Palimpzest semantic aggregation is temporarily disabled due to validation issues.")
        # if not hasattr(pz, 'MemoryDataset'):
        #     raise NotImplementedError("Palimpzest does not fully support semantic aggregation operation directly.")
        # try:
        #     df_palimpzest, column_mapping = self.prepare_dataframe(df)

        #     # Use temporary environment variable for API key
        #     with temporary_env_var("OPENAI_API_KEY", self.api_key):
        #         pz_dataset = pz.MemoryDataset(id="temp_dataset", vals=df_palimpzest)
        #         pz_dataset = pz_dataset.sem_agg(
        #             col={'name': alias, 'type': str, 'desc': user_prompt},
        #             agg=user_prompt,
        #             # depends_on=column
        #         )
        #         output = pz_dataset.run(config=pz.QueryProcessorConfig(
        #             # default policy is MaxQuality
        #             available_models=[self.model],
        #         ))
        #         result_df = output.to_df()
        #     logging.info(f"Result of Palimpzest SEM_AGG: {result_df.head(10)}")
        #     if not result_df.empty and column_mapping:
        #         result_df = self.restore_dataframe(result_df, column_mapping)

        #     logging.info(f"Result of Palimpzest SEM_AGG: {result_df.head(10)}")
        #     return result_df
        # except Exception as e:
        #     logging.error(f"Error during Palimpzest SEM_AGG: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return df
    
    def sem_distinct(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Semantic deduplication using Palimpzest - falls back to standard deduplication."""
        raise NotImplementedError("Palimpzest does not support semantic distinct operation directly.")
    
    def sem_order_by(self, df: pd.DataFrame, user_prompt: str, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic ordering using Palimpzest."""
        num_of_records = len(df)
        specified_column = column
        if specified_column is None:
            raise ValueError("Column name must be specified for Palimpzest retrieve operation")
        relevant_col_name = f"relevant_{specified_column}"

        def search_func(index: chromadb.Collection, query, k):
            # results = index.query(query, n_results=k)
            results = index.query(query_texts=[user_prompt], n_results=k)

            final_results = []
            for doc_list in results["documents"]:
                for doc in doc_list:
                    final_results.append(doc)
            
            return {relevant_col_name: final_results}
        
        try:
            self._reset_api_call_count()
            relevant_columns = [
                {"name": relevant_col_name, "type": list[str], "desc": f"Relevant {specified_column} based on {user_prompt}"}
            ]
            env_vars = {}
            # Use temporary environment variable for API key
            if self.embedding_model.startswith("litellm_proxy"):
                env_vars = {
                    "OPENAI_API_KEY": "-",
                    "OPENAI_BASE_URL": self.embedding_api_base
                }
                embedding_function = OpenAIEmbeddingFunction(
                    api_key="-",
                    api_base=self.embedding_api_base,
                    model_name=self.embedding_model
                )
                # OPENAI_BASE_URL = self.embedding_api_base
                # OPENAI_API_KEY = "-"
            elif self.api_key in ('dummy', None):
                embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            else:
                env_vars = {
                    "OPENAI_API_KEY": self.api_key,
                }
                embedding_function = OpenAIEmbeddingFunction(
                    api_key=self.api_key,
                    model_name=self.embedding_model
                )
            
            embedding_function = self._wrap_embedding_function(embedding_function)
            
            with temporary_env_vars(env_vars):
                index = self.chroma_client.create_collection(
                    name="palimpzest_collection", 
                    embedding_function=embedding_function
                )
                index.add(
                    documents=df[specified_column].tolist(),
                    ids=[f"id{i}" for i in range(1, num_of_records + 1)]
                )
                if hasattr(pz, 'MemoryDataset'):
                    pz_dataset = pz.MemoryDataset(id="temp_dataset", vals=df)
                    pz_dataset = pz_dataset.sem_topk(
                        index=index,
                        search_func=search_func,
                        # search_attr='_search_query',
                        search_attr=specified_column,
                        output_attrs=relevant_columns,
                        k=num_of_records,
                    )
                else:
                    pz_dataset = pz.Dataset(source=df)
                    pz_dataset = pz_dataset.retrieve(index=index,
                                                            search_func=search_func,
                                                            # search_attr='_search_query',
                                                            search_attr=specified_column,
                                                            output_attrs=relevant_columns,
                                                            k=num_of_records,
                    )
                output = pz_dataset.run(config=pz.QueryProcessorConfig(
                    # default policy is MaxQuality
                    available_models=[self.model],
                    api_base=self.api_base,
                ))

            index.delete(ids=[f"id{i}" for i in range(1, num_of_records + 1)])
            result_df = output.to_df()
            # Extract stats
            if hasattr(output, 'execution_stats'):
                self.last_stats = extract_palimpzest_stats(output.execution_stats)
            
            if relevant_col_name in result_df.columns:
                # Get the relevant list
                first_relevant_list = result_df[relevant_col_name].iloc[0]
                
                # Handle case where relevant_list is a string representation of a list
                if isinstance(first_relevant_list, str):
                    import ast
                    try:
                        first_relevant_list = ast.literal_eval(first_relevant_list)
                    except (ValueError, SyntaxError):
                        first_relevant_list = [first_relevant_list]
                
                # Create mapping from item value to position in relevant list
                position_map = {}
                if isinstance(first_relevant_list, list):
                    for pos, item in enumerate(first_relevant_list):
                        item_str = str(item)
                        if item_str not in position_map:
                            position_map[item_str] = pos
                
                # Assign positions to each row
                new_indices = []
                for idx, row in result_df.iterrows():
                    current_value = str(row[specified_column])
                    
                    if current_value in position_map:
                        position = position_map[current_value]
                    else:
                        # If not found in relevant list, assign based on row index
                        position = len(first_relevant_list) + idx
                    
                    new_indices.append(position)
                
                # Replace the list column with integer indices
                result_df[relevant_col_name] = new_indices

                # Replace the column name with '_relevant'
                result_df = result_df.rename(columns={relevant_col_name: '_relevant'})
                # Sort by the new '_relevant' column
                result_df = result_df.sort_values(by=['_relevant']).reset_index(drop=True)

            self.last_stats.api_calls = self._get_api_call_count()
            self.last_stats.embedding_calls = self._get_embedding_call_count()
            logging.info(f"Result of Palimpzest SEM_ORDER_BY: {result_df.head(10)}")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_ORDER_BY: {e}")
            import traceback
            traceback.print_exc()
            return df

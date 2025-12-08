"""
LOTUS backend implementation for SABER semantic operations.
"""
import os
import re
import time
import threading
import shutil
from typing import Dict, Optional, List
import logging
import pandas as pd
import litellm

from .base_backend import BaseBackend
from ..config import LOTUS_DEFAULT_DEDUP_THRESHOLD
from ..benchmark import BenchmarkStats, extract_lotus_stats

logger = logging.getLogger(__name__)

class LOTUSBackend(BaseBackend):
    """LOTUS backend for semantic operations using embedding-based similarity."""
    
    def __init__(self, api_key: str = None, model: str = None, api_base: str = None, embedding_model: str = None, embedding_api_base: str = None):
        super().__init__("LOTUS", api_key, model, api_base, embedding_model, embedding_api_base)
        self.last_stats = BenchmarkStats()
        self._lm = None  # Will be set by engine
        self._api_call_count = 0
        self._embedding_call_count = 0
        self._api_call_lock = threading.Lock()
        self._register_callbacks()
        self._rm_wrapped = False
    
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
    
    def _wrap_rm_for_tracking(self):
        """Wrap the retrieval model to track embedding calls."""
        import lotus
        from lotus.models import SentenceTransformersRM, LiteLLMRM
        
        if self._rm_wrapped or not hasattr(lotus.settings, 'rm') or lotus.settings.rm is None:
            return
        
        rm = lotus.settings.rm
        
        # For LiteLLMRM, callbacks already handle tracking
        # For SentenceTransformersRM, we need to wrap the call
        if isinstance(rm, SentenceTransformersRM):
            original_call = rm.__call__
            backend = self
            
            def tracked_call(docs):
                with backend._api_call_lock:
                    backend._embedding_call_count += 1
                return original_call(docs)
            
            rm.__call__ = tracked_call
            self._rm_wrapped = True
        elif isinstance(rm, LiteLLMRM):
            # LiteLLMRM uses litellm callbacks, already tracked
            self._rm_wrapped = True
    
    def _track_operation(self, operation_func, *args, **kwargs):
        """Track cost and latency for a LOTUS operation."""
        # Wrap RM for tracking even if _lm is None (for embedding-only operations)
        if not self._rm_wrapped:
            self._wrap_rm_for_tracking()
        
        if self._lm is None:
            # If LM is not set, track embeddings only
            self._reset_api_call_count()
            start_time = time.time()
            
            try:
                result = operation_func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                
                self.last_stats = BenchmarkStats(
                    total_semantic_execution_time_seconds=elapsed_time,
                    api_calls=self._get_api_call_count(),
                    embedding_calls=self._get_embedding_call_count()
                )
                
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                self.last_stats = BenchmarkStats(
                    total_semantic_execution_time_seconds=elapsed_time,
                    api_calls=self._get_api_call_count(),
                    embedding_calls=self._get_embedding_call_count()
                )
                raise
        
        # Reset LM stats and API counter before operation
        self._lm.reset_stats()
        self._reset_api_call_count()
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # Extract stats from LM
            self.last_stats = extract_lotus_stats(self._lm)
            self.last_stats.total_semantic_execution_time_seconds = elapsed_time
            self.last_stats.api_calls = self._get_api_call_count()
            self.last_stats.embedding_calls = self._get_embedding_call_count()
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.last_stats = BenchmarkStats(
                total_semantic_execution_time_seconds=elapsed_time,
                api_calls=self._get_api_call_count(),
                embedding_calls=self._get_embedding_call_count()
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
        """Update prompts to use underscore column names for LOTUS.
        
        Also handles table alias prefixes (e.g., {d.biography} -> {biography})
        """
        updated_prompt = prompt
        
        # First, handle table alias prefixes in column references
        # Pattern matches {alias.column} or {alias.column_name}
        alias_pattern = r'\{([a-zA-Z_]\w*)\.([a-zA-Z_][\w]*)\}'
        def replace_alias(match):
            # Extract just the column part, ignoring the alias
            return '{' + match.group(2) + '}'
        updated_prompt = re.sub(alias_pattern, replace_alias, updated_prompt)
        
        # Then replace dot notation with underscore notation in prompts
        for underscore_col, original_col in column_mapping.items():
            # Replace {original.col} with {underscore_col}
            pattern = r'\{' + re.escape(original_col) + r'\}'
            replacement = '{' + underscore_col + '}'
            updated_prompt = re.sub(pattern, replacement, updated_prompt)
        
        return updated_prompt
    
    def _validate_column_references(self, prompt: str, df: pd.DataFrame, operation_name: str = "semantic operation") -> str:
        """Validate that all column references in prompt exist in the DataFrame."""
        # Extract column references from prompt: {column_name} pattern
        column_refs = re.findall(r'\{([^}]+)\}', prompt)
        
        if not column_refs:
            # No column references found - LOTUS will reject this
            # Auto-add the most appropriate column reference
            logger.warning(f"{operation_name}: No column references found in prompt, will add fallback")
            
            # Find text columns as candidates
            text_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'string']
            if not text_cols:
                # Fallback to any column
                text_cols = list(df.columns)
            
            if text_cols:
                # Prepend column reference to prompt
                fallback_prompt = f"{{{text_cols[0]}}} {prompt}"
                logger.info(f"{operation_name}: Added fallback column reference: {fallback_prompt[:100]}")
                return fallback_prompt
            else:
                logger.warning(f"{operation_name}: No columns available for fallback")
                return prompt
        
        available_cols = set(df.columns)
        missing_cols = []
        
        for col_ref in column_refs:
            # Clean column reference: remove backticks, quotes, and extra spaces
            clean_ref = col_ref.strip().replace('`', '').replace('"', '').replace("'", '')
            
            # Handle table.column references
            if '.' in clean_ref or ':' in clean_ref:
                # Extract just the column name part
                parts = re.split(r'[.:]', clean_ref)
                col_name = parts[-1].strip() if parts else clean_ref
            else:
                col_name = clean_ref
            
            # Case-insensitive check
            if not any(col_name.lower() == avail_col.lower() for avail_col in available_cols):
                missing_cols.append(col_ref)
        
        if missing_cols:
            # Attempt to resolve missing columns
            resolved_prompt = prompt
            still_missing = []
            
            for missing_ref in missing_cols:
                # Clean the reference again
                clean_ref = missing_ref.strip().replace('`', '').replace('"', '').replace("'", '')
                if '.' in clean_ref:
                    col_name = clean_ref.split('.')[-1].strip()
                else:
                    col_name = clean_ref
                
                # Strategy 1: Check if the column name exists without table prefix
                # (e.g. {table.col} -> {col})
                if any(col_name.lower() == avail_col.lower() for avail_col in available_cols):
                    # Find the exact match in available columns
                    matched_col = next(avail_col for avail_col in available_cols if col_name.lower() == avail_col.lower())
                    pattern = r'\{' + re.escape(missing_ref) + r'\}'
                    replacement = '{' + matched_col + '}'
                    resolved_prompt = re.sub(pattern, replacement, resolved_prompt)
                    logger.info(f"{operation_name}: Resolved {missing_ref} to {{{matched_col}}}")
                    continue
                
                # Strategy 2: Check for suffix match (e.g. {long_table_name_col} -> {col})
                # or {col} -> {table_col}
                suffix_match = None
                for avail_col in available_cols:
                    if avail_col.lower().endswith(col_name.lower()) or col_name.lower().endswith(avail_col.lower()):
                        # Only accept if it's a significant match (len > 3) to avoid false positives
                        if len(col_name) > 3 and len(avail_col) > 3:
                            suffix_match = avail_col
                            break
                
                if suffix_match:
                    pattern = r'\{' + re.escape(missing_ref) + r'\}'
                    replacement = '{' + suffix_match + '}'
                    resolved_prompt = re.sub(pattern, replacement, resolved_prompt)
                    logger.info(f"{operation_name}: Fuzzy resolved {missing_ref} to {{{suffix_match}}}")
                    continue
                
                still_missing.append(missing_ref)
            
            if not still_missing:
                return resolved_prompt
                
            # Fallback: if only one column available, replace missing references with it
            if len(available_cols) == 1:
                only_col = list(available_cols)[0]
                modified_prompt = resolved_prompt
                for missing_col in still_missing:
                    pattern = r'\{' + re.escape(missing_col) + r'\}'
                    replacement = '{' + only_col + '}'
                    modified_prompt = re.sub(pattern, replacement, modified_prompt)
                logger.warning(
                    f"{operation_name}: Replaced missing column references {still_missing} "
                    f"with only available column '{only_col}'"
                )
                return modified_prompt
            
            raise ValueError(
                f"Column reference error in {operation_name}: "
                f"Columns {still_missing} not found in DataFrame. "
                f"Available columns: {sorted(list(available_cols))}. "
                f"The prompt contains invalid column references that don't exist in the table. "
                f"Check for typos, backticks, or incorrect column names."
            )
        
        return prompt
    
    def sem_where(self, df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
        """Semantic filtering using LOTUS."""
        def _operation():
            df_lotus, column_mapping = self.prepare_dataframe(df)
            updated_prompt = self.update_prompt(user_prompt, column_mapping)
            
            # Check if we have column placeholders - if not, try to add them intelligently
            if '{' not in updated_prompt:
                # Try to detect which columns might be referenced
                col_candidates = []
                for col in df_lotus.columns:
                    # Check if column name appears in prompt (case-insensitive, check original name too)
                    original_col = column_mapping.get(col, col)
                    if col.lower() in updated_prompt.lower() or original_col.lower() in updated_prompt.lower():
                        col_candidates.append(col)
                
                if col_candidates:
                    # Use the first matching column
                    logger.info(f"Auto-adding column reference {{{col_candidates[0]}}} to SEM_WHERE prompt")
                    updated_prompt = f"{{{col_candidates[0]}}} {updated_prompt}"
                else:
                    # Default to first text column if available
                    text_cols = [col for col in df_lotus.columns if df_lotus[col].dtype == 'object']
                    if text_cols:
                        logger.warning(f"No column references found. Using first text column {{{text_cols[0]}}}")
                        updated_prompt = f"{{{text_cols[0]}}} {updated_prompt}"
            
            # Validate column references before executing
            updated_prompt = self._validate_column_references(updated_prompt, df_lotus, "SEM_WHERE")
            
            result_df_lotus = df_lotus.sem_filter(updated_prompt)
            result_df = self.restore_dataframe(result_df_lotus, column_mapping)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns)
            
            logging.info(f"Result of LOTUS SEM_WHERE: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except ValueError as e:
            # Column reference error - provide clear message
            error_msg = str(e)
            if "no parameterized columns" in error_msg.lower():
                logging.error(f"LOTUS SEM_WHERE error: Prompt must include column references using {{column_name}}. "
                            f"Prompt: '{user_prompt}'")
                raise ValueError(f"LOTUS SEM_WHERE error: Prompt must include column references using {{column_name}}. "
                               f"Prompt: '{user_prompt}'") from e
            logging.error(f"Column reference error in LOTUS SEM_WHERE: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_WHERE: {e}")
            raise RuntimeError(f"LOTUS SEM_WHERE execution failed: {str(e)}") from e
    
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection using LOTUS."""
        # Ensure alias doesn't conflict with existing columns
        unique_alias = self._ensure_unique_alias(df, alias)
        
        def _operation():
            df_lotus, column_mapping = self.prepare_dataframe(df)
            updated_prompt = self.update_prompt(user_prompt, column_mapping)
            
            # Check if we have column placeholders for extraction operations
            if '{' not in updated_prompt and ('extract' in updated_prompt.lower() or 'get' in updated_prompt.lower()):
                # Try to detect which columns might be referenced
                for col in df_lotus.columns:
                    original_col = column_mapping.get(col, col)
                    if col.lower() in updated_prompt.lower() or original_col.lower() in updated_prompt.lower():
                        logger.info(f"Auto-adding column reference {{{col}}} to SEM_SELECT prompt")
                        updated_prompt = f"Extract from {{{col}}}: {updated_prompt}"
                        break
            
            # Validate column references before executing
            updated_prompt = self._validate_column_references(updated_prompt, df_lotus, "SEM_SELECT")
            
            result_df_lotus = df_lotus.sem_map(updated_prompt, suffix=unique_alias)
            result_df = self.restore_dataframe(result_df_lotus, column_mapping)

            # Validate that the new column was created and has content
            if unique_alias not in result_df.columns:
                raise ValueError(
                    f"SEM_SELECT failed to create column '{unique_alias}'. "
                    f"Check that prompt is specific and complete. "
                    f"Example: 'Extract country code from {{column}} as 2-letter code' "
                    f"instead of just 'Extract'."
                )
            
            # Check if all values are NULL/empty and provide actionable feedback
            null_count = result_df[unique_alias].isna().sum()
            total_count = len(result_df)
            if null_count == total_count and total_count > 0:
                logger.warning(
                    f"SEM_SELECT column '{unique_alias}' contains only NULL values ({total_count}/{total_count}). "
                    f"This likely means the extraction prompt is unclear or too restrictive. "
                    f"Prompt was: {updated_prompt[:100]}... "
                    f"Suggestion: Be more specific about what to extract and the expected format."
                )
            elif null_count > total_count * 0.5:
                logger.warning(
                    f"SEM_SELECT column '{unique_alias}' has {null_count}/{total_count} NULL values. "
                    f"Many rows failed extraction. Consider simplifying the prompt or checking data format."
                )

            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns.tolist() + [unique_alias])
            
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
        
        # Clean up any existing index directory to avoid stale data
        index_name = f"{column.replace(' ', '_')}_index"
        if os.path.exists(index_name):
            shutil.rmtree(index_name)
            logging.info(f"Cleaned up existing index directory: {index_name}")
        
        def _operation():
            result_df = df.sem_index(column, index_name).sem_cluster_by(column, number_of_groups)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns)
            
            logging.info(f"Result of LOTUS SEM_GROUP_BY: {result_df.head(10)}")
            return result_df
        
        try:
            result = self._track_operation(_operation)
            
            # Clean up index directory after operation
            if os.path.exists(index_name):
                shutil.rmtree(index_name)
                logging.info(f"Cleaned up index directory after operation: {index_name}")
            
            return result
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_GROUP_BY: {e}")
            # Clean up on error too
            if os.path.exists(index_name):
                shutil.rmtree(index_name)
            return df
    
    def sem_agg(self, df: pd.DataFrame, user_prompt: str, alias: str,
               group_by_col: Optional[List[str]] = None, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic aggregation using LOTUS."""
        def _operation():
            if group_by_col:
                result_df = df.sem_agg(user_prompt, suffix=alias, group_by=group_by_col)
            else:
                result_df = df.sem_agg(user_prompt, suffix=alias)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                expected_cols = group_by_col if group_by_col else []
                result_df = pd.DataFrame(columns=expected_cols + [alias])
            
            logging.info(f"Result of LOTUS SEM_AGG: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_AGG: {e}")
            return df
    
    def sem_distinct(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Semantic deduplication using LOTUS."""
        # Clean up any existing index directory to avoid stale data
        index_name = f"{column}_index"
        if os.path.exists(index_name):
            shutil.rmtree(index_name)
            logging.info(f"Cleaned up existing index directory: {index_name}")
        
        def _operation():
            result_df = df.sem_index(f"{column}", index_name).sem_dedup(f"{column}", threshold=LOTUS_DEFAULT_DEDUP_THRESHOLD)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns)
            
            logging.info(f"Result of LOTUS SEM_DISTINCT: {result_df.head(10)}")
            return result_df
        
        try:
            result = self._track_operation(_operation)
            
            # Clean up index directory after operation
            if os.path.exists(index_name):
                shutil.rmtree(index_name)
                logging.info(f"Cleaned up index directory after operation: {index_name}")
            
            return result
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_DISTINCT: {e}")
            # Clean up on error too
            if os.path.exists(index_name):
                shutil.rmtree(index_name)
            return df
    
    def sem_order_by(self, df: pd.DataFrame, user_prompt: str, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic ordering using LOTUS."""
        def _operation():
            number_of_records = df.shape[0]
            if number_of_records == 0:
                logging.info("DataFrame is empty, returning original DataFrame.")
                return df
            result_df = df.sem_topk(user_prompt, K=number_of_records, return_stats=False)
            
            # Ensure columns are preserved even when empty
            if result_df.empty and not df.empty:
                result_df = pd.DataFrame(columns=df.columns)
            
            logging.info(f"Result of LOTUS SEM_ORDER_BY: {result_df.head(10)}")
            return result_df
        
        try:
            return self._track_operation(_operation)
        except Exception as e:
            logging.error(f"Error during LOTUS SEM_ORDER_BY: {e}")
            return df

"""
Palimpzest backend implementation for SABER semantic operations.
"""
import re
import os
from contextlib import contextmanager
from typing import Dict, Optional, List
import logging
import pandas as pd

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

    def sem_where(self, df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
        """Semantic filtering using Palimpzest."""
        try:
            df_palimpzest, column_mapping = self.prepare_dataframe(df)

            # Use temporary environment variable for API key
            with temporary_env_var("OPENAI_API_KEY", self.api_key):
                if hasattr(pz, 'MemoryDataset'):
                    pz_dataset = pz.MemoryDataset(id="temp_dataset", vals=df_palimpzest)
                else:
                    pz_dataset = pz.Dataset(source=df_palimpzest)
                pz_dataset = pz_dataset.sem_filter(user_prompt)
                output = pz_dataset.run(config=pz.QueryProcessorConfig(
                    # default policy is MaxQuality
                    available_models=[self.model],
                ))
                result_df = output.to_df()
                
                # Extract stats
                if hasattr(output, 'execution_stats'):
                    self.last_stats = extract_palimpzest_stats(output.execution_stats)

            if not result_df.empty and column_mapping:
                result_df = self.restore_dataframe(result_df, column_mapping)

            logging.info(f"Result of Palimpzest SEM_WHERE: {result_df.head(10)}")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_WHERE: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection using Palimpzest."""
        try:
            df_palimpzest, column_mapping = self.prepare_dataframe(df)

            # Use temporary environment variable for API key
            with temporary_env_var("OPENAI_API_KEY", self.api_key):
                if hasattr(pz, 'MemoryDataset'):
                    pz_dataset = pz.MemoryDataset(id="temp_dataset", vals=df_palimpzest)
                    pz_dataset = pz_dataset.sem_map([{"name": alias, "type": str, "desc": user_prompt}])
                else:
                    pz_dataset = pz.Dataset(source=df_palimpzest)
                    pz_dataset = pz_dataset.sem_add_columns([{"name": alias, "type": str, "desc": user_prompt}])
                output = pz_dataset.run(config=pz.QueryProcessorConfig(
                    # default policy is MaxQuality
                    available_models=[self.model],
                ))
                result_df = output.to_df()
                
                # Extract stats
                if hasattr(output, 'execution_stats'):
                    self.last_stats = extract_palimpzest_stats(output.execution_stats)

            if not result_df.empty and column_mapping:
                result_df = self.restore_dataframe(result_df, column_mapping)

            logging.info(f"Result of Palimpzest SEM_SELECT: {result_df.head(10)}")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_SELECT: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def sem_join(self, df1: pd.DataFrame, df2: pd.DataFrame, user_prompt: str,
                df1_name: str = "left", df2_name: str = "right") -> pd.DataFrame:
        """Semantic join using Palimpzest."""
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
        if not hasattr(pz, 'MemoryDataset'):
            raise NotImplementedError("Palimpzest does not support semantic join operation directly.")
        try:
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
                ))
                result_df = output.to_df()
            
            # Extract stats
            if hasattr(output, 'execution_stats'):
                self.last_stats = extract_palimpzest_stats(output.execution_stats)
            result_df = rename_sem_join_columns(result_df, df1_name, df2_name)

            if not result_df.empty and column_mapping:
                result_df = self.restore_dataframe(result_df, column_mapping)

            logging.info(f"Result of Palimpzest SEM_JOIN: {result_df.head(10)}")
            return result_df
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

            logging.info(f"Result of Palimpzest SEM_ORDER_BY: {result_df.head(10)}")
            return result_df
        except Exception as e:
            logging.error(f"Error during Palimpzest SEM_ORDER_BY: {e}")
            import traceback
            traceback.print_exc()
            return df

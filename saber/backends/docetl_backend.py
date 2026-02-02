"""
DocETL backend implementation for SABER semantic operations.
"""
import os
import re
import subprocess
import sys
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import logging
import pandas as pd
import litellm

from .base_backend import BaseBackend
from ..config import DOCETL_DEFAULT_MODEL, DOCETL_DEFAULT_EMBEDDING_MODEL, DOCETL_DEFAULT_RESOLVE_THRESHOLD, MAX_TOKENS
from ..benchmark import BenchmarkStats, extract_docetl_stats

logger = logging.getLogger(__name__)

class DocETLBackend(BaseBackend):
    """DocETL backend for semantic operations using LLM-based document processing."""
    
    def __init__(self, api_key: str = None, model: str = None, api_base: str = None, embedding_model: str = None, embedding_api_base: str = None):
        super().__init__("docetl", api_key, model, api_base, embedding_model, embedding_api_base)
        # Set default model if not provided
        if self.model is None:
            self.model = DOCETL_DEFAULT_MODEL
        if self.embedding_model is None:
            self.embedding_model = DOCETL_DEFAULT_EMBEDDING_MODEL
        self.last_stats = BenchmarkStats()
    
    def _get_output_mode(self) -> str:
        """Determine output mode based on function calling support.
        
        Returns 'tools' for models that support function calling, 'structured_output' otherwise.
        """
        if litellm.supports_function_calling(self.model):
            return 'tools'
        else:
            return 'structured_output'
    
    def prepare_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
        """Replace dots in column names with underscores for DocETL."""
        column_mapping = {}
        new_columns = []
        
        for col in df.columns:
            if '.' in col:
                new_col = col.replace('.', '_')
                column_mapping[new_col] = col
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        
        df_docetl = df.copy()
        df_docetl.columns = new_columns
        
        return df_docetl, column_mapping
    
    def restore_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Restore original column names after DocETL processing."""
        df_restored = df.copy()
        
        new_columns = []
        for col in df_restored.columns:
            original_col = column_mapping.get(col, col)
            new_columns.append(original_col)
        
        df_restored.columns = new_columns
        return df_restored
    
    def update_prompt(self, prompt: str, column_mapping: Dict[str, str]) -> str:
        """Update prompts to use underscore column names for DocETL.
        
        Handles table alias prefixes in joined tables:
        - {{ input.c.experience }} -> {{ input.c_experience }} (for joined table columns like c_experience)
        - {{ input.r.description }} -> {{ input.r_description }} (for joined table columns like r_description)
        
        The column_mapping maps underscore names back to dot names (e.g., 'c_experience' -> 'c.experience').
        For prompts, we need to reverse this: convert dot notation in prompts to underscore notation
        that matches the actual DataFrame column names.
        """
        updated_prompt = prompt
        
        # Handle table alias prefixes in DocETL Jinja2 templates
        # Pattern matches {{ input.alias.column }} and converts to {{ input.alias_column }}
        # This handles joined table columns where columns are named like "r.name", "c.experience"
        # and need to be referenced as "r_name", "c_experience" in DocETL
        alias_pattern = r'\{\{\s*input\.([a-zA-Z_]\w*)\.([a-zA-Z_][\w]*)\s*\}\}'
        def replace_docetl_alias(match):
            alias = match.group(1)
            column = match.group(2)
            # Combine alias and column with underscore to match DataFrame column names
            return '{{ input.' + alias + '_' + column + ' }}'
        updated_prompt = re.sub(alias_pattern, replace_docetl_alias, updated_prompt)
        
        # Also handle {{ left.alias.column }} and {{ right.alias.column }} for joins
        left_alias_pattern = r'\{\{\s*left\.([a-zA-Z_]\w*)\.([a-zA-Z_][\w]*)\s*\}\}'
        def replace_left_alias(match):
            alias = match.group(1)
            column = match.group(2)
            return '{{ left.' + alias + '_' + column + ' }}'
        updated_prompt = re.sub(left_alias_pattern, replace_left_alias, updated_prompt)
        
        right_alias_pattern = r'\{\{\s*right\.([a-zA-Z_]\w*)\.([a-zA-Z_][\w]*)\s*\}\}'
        def replace_right_alias(match):
            alias = match.group(1)
            column = match.group(2)
            return '{{ right.' + alias + '_' + column + ' }}'
        updated_prompt = re.sub(right_alias_pattern, replace_right_alias, updated_prompt)
        
        # Then replace dot notation with underscore notation in DocETL templates
        # This handles direct column references like {{ input.r.name }} that were already converted above,
        # and also handles any remaining cases from column_mapping
        for underscore_col, original_col in column_mapping.items():
            # Replace {{ input.original.col }} with {{ input.underscore_col }}
            pattern = r'\{\{\s*input\.' + re.escape(original_col) + r'\s*\}\}'
            replacement = '{{ input.' + underscore_col + ' }}'
            updated_prompt = re.sub(pattern, replacement, updated_prompt)
            
            # Also handle other template patterns that might use the column name
            # Replace {original.col} with {underscore_col} (for lotus-style references)
            pattern = r'\{' + re.escape(original_col) + r'\}'
            replacement = '{' + underscore_col + '}'
            updated_prompt = re.sub(pattern, replacement, updated_prompt)
        
        return updated_prompt
    
    def _get_operation_config(self, operation_type: str, user_prompt: str, alias: str = None, **kwargs) -> dict:
        """Get operation-specific configuration."""
        if operation_type == 'join':
            return {
                'name': 'join_semantic',
                'type': 'equijoin',
                'comparison_prompt': user_prompt,
            }
        elif operation_type == 'filter':
            return {
                'name': 'filter_where',
                'type': 'filter',
                'prompt': user_prompt,
                'output': {
                    'schema': {
                        'matches_criteria': 'boolean'
                    },
                    'mode': self._get_output_mode()
                }
            }
        elif operation_type == 'cluster':
            column = kwargs.get('column', 'name')
            number_of_groups = kwargs.get('number_of_groups', 3)
            
            return {
                'name': 'cluster_semantic',
                'type': 'cluster',
                'embedding_keys': [column],
                'max_batch_size': min(number_of_groups, 5),
                'output_key': 'cluster_info',
                'summary_schema': {
                    'concept': 'string',
                    'description': 'string'
                },
                'embedding_model': self.embedding_model,
                'summary_prompt': f'''You are analyzing a cluster of {column} values. Create a representative name and description for this semantic group.

Items in this cluster:
{{% for input in inputs %}}
- {{{{ input.{column} }}}}
{{% endfor %}}

Provide:
1. concept: a short, descriptive name that represents all items in this cluster
2. description: a brief explanation of what unites these items

Make the concept name specific enough to distinguish this group from others, but general enough to encompass all items.'''
            }
        elif operation_type == 'reduce':
            reduce_key = kwargs.get('reduce_key', 'cluster_id')
            
            return {
                'name': 'reduce_semantic',
                'type': 'reduce',
                'reduce_key': [reduce_key] if isinstance(reduce_key, str) else reduce_key,
                'pass_through': True,
                'output': {
                    'schema': {
                        # reduce_key: 'string',
                        alias: 'string'
                    },
                    'mode': self._get_output_mode()
                },
                'prompt': user_prompt
            }
        elif operation_type == 'select':
            return {
                'name': 'map_select',
                'type': 'map',
                'prompt': user_prompt,
                'output': {
                    'schema': {
                        alias: 'string'
                    },
                    'mode': self._get_output_mode()
                }
            }
        elif operation_type == 'resolve':
            column = kwargs.get('column', 'name')
            threshold = kwargs.get('threshold', 0.6)
            return {
                'name': 'resolve_distinct',
                'type': 'resolve',
                'optimize': True,
                'blocking_keys': [column],
                'blocking_threshold': threshold,
                'embedding_model': self.embedding_model,
                'comparison_prompt': f"""Compare these two {column} entries for semantic similarity:

Entry 1: {{{{ input1.{column} }}}}
Entry 2: {{{{ input2.{column} }}}}

Consider:
- Similar meanings despite different wordings
- Common abbreviations and variations
- Typos and formatting differences
- Synonyms and alternative names

Are these entries semantically equivalent (representing the same concept)?
Respond with "True" if they are duplicates, "False" if they are distinct.""",
                'resolution_prompt': f"""Given the following identified multiple {column} entries that are semantic duplicates:
{{% for entry in inputs %}}
- {{{{ entry.{column} }}}}
{{% endfor %}}

Select the BEST representative entry from the above list. Choose the one that is:
1. Most complete and informative
2. Uses standard terminology
3. Has proper formatting
4. Would be most widely recognized

Return only the selected {column} value, nothing else.""",
                'output': {
                    'schema': {
                        column: 'string'
                    },
                    'mode': self._get_output_mode()
                }
            }
        elif operation_type == 'rank':
            direction = kwargs.get('direction', 'desc')
            input_keys = kwargs.get('input_keys', None)
            call_budget = kwargs.get('call_budget', 10)
            initial_ordering_method = kwargs.get('initial_ordering_method', 'likert')
            
            # If no input_keys specified, use all columns except id columns
            if input_keys is None:
                input_keys = [col for col in kwargs.get('df_columns', []) 
                            if not col.lower().endswith('id') and col.lower() != 'id']
            
            return {
                'name': 'rank_semantic',
                'type': 'rank',
                'prompt': user_prompt,
                'input_keys': input_keys,
                'direction': direction,
                'rerank_call_budget': call_budget,
                'initial_ordering_method': initial_ordering_method
            }
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def _get_pipeline_step(self, operation_type: str) -> dict:
        """Get pipeline step configuration."""
        if operation_type == 'join':
            return {
                'name': 'join_step',
                'operations': [
                    {
                        'join_semantic': {
                            'left': 'input_data1',
                            'right': 'input_data2',
                        }
                    }
                ]
            }
        elif operation_type == 'filter':
            return {
                'name': 'filter_step',
                'input': 'input_data1',
                'operations': ['filter_where']
            }
        elif operation_type == 'cluster':
            return {
                'name': 'cluster_step',
                'input': 'input_data1',
                'operations': ['cluster_semantic']
            }
        elif operation_type == 'reduce':
            return {
                'name': 'reduce_step',
                'input': 'input_data1',
                'operations': ['reduce_semantic']
            }
        elif operation_type == 'select':
            return {
                'name': 'select_step',
                'input': 'input_data1',
                'operations': ['map_select']
            }
        elif operation_type == 'resolve':
            return {
                'name': 'resolve_step',
                'input': 'input_data1',
                'operations': ['resolve_distinct']
            }
        elif operation_type == 'rank':
            return {
                'name': 'rank_step',
                'input': 'input_data1',
                'operations': ['rank_semantic']
            }
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def _process_cluster_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DocETL cluster output to match expected format."""
        try:
            logging.info(f"Processing cluster output. DataFrame shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")

            # Check the output_key from the configuration (default is 'cluster_info')
            cluster_column = 'cluster_info'
            if cluster_column not in df.columns:
                # Fallback to other possible column names
                possible_columns = [col for col in df.columns if 'cluster' in col.lower() or 'categories' in col.lower()]
                if possible_columns:
                    cluster_column = possible_columns[0]
                else:
                    # logging.info("No cluster column found, adding default cluster_id")
                    # df['cluster_id'] = 'cluster_001'
                    # return df
                    raise ValueError("No cluster information column found in DocETL output.")
            
            cluster_ids = []
            cluster_names = []
            
            for idx, row in df.iterrows():
                cluster_info = row[cluster_column]
                
                if isinstance(cluster_info, list) and len(cluster_info) > 0:
                    # DocETL returns hierarchical clusters - use the most specific (first) one
                    most_specific_cluster = cluster_info[0]
                    
                    if isinstance(most_specific_cluster, dict):
                        # Extract cluster information
                        cluster_id = most_specific_cluster.get('concept', f'cluster_{idx+1:03d}')
                        cluster_name = most_specific_cluster.get('concept', cluster_id)
                    else:
                        cluster_id = f'cluster_{idx+1:03d}'
                        cluster_name = str(most_specific_cluster)
                        
                elif isinstance(cluster_info, dict):
                    cluster_id = cluster_info.get('concept', f'cluster_{idx+1:03d}')
                    cluster_name = cluster_info.get('concept', cluster_id)
                else:
                    cluster_id = f'cluster_{idx+1:03d}'
                    cluster_name = f'Group {idx+1}'
                
                cluster_ids.append(cluster_id)
                cluster_names.append(cluster_name)
            
            # Add cluster information
            df['cluster_id'] = cluster_ids
            df['cluster_name'] = cluster_names

            logging.info(f"Unique cluster IDs found: {df['cluster_id'].unique()}")
            logging.info(f"Cluster distribution: {df['cluster_id'].value_counts()}")
            logging.info(f"Sample cluster names: {df['cluster_name'].unique()[:5]}")

            return df
            
        except Exception as e:
            logging.error(f"Error processing cluster output: {e}")
            # Fallback: assign sequential cluster IDs
            df['cluster_id'] = [f'cluster_{i+1:03d}' for i in range(len(df))]
            df['cluster_name'] = [f'Group {i+1}' for i in range(len(df))]
            return df
    
    def _run_operation(self, operation_type: str, df1: pd.DataFrame, 
                      user_prompt: str, alias: str = None, df2: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """Generic DocETL operation runner via CLI."""
        try:
            if kwargs.get('reduce_key', None) and kwargs['reduce_key'] not in df1.columns:
                raise ValueError(f"Reduce key '{kwargs['reduce_key']}' not found in DataFrame columns.")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Store dataframe(s) as JSON files
                input_file1 = temp_path / "input_data1.json"
                df1.to_json(input_file1, orient='records', indent=2)
                
                datasets = {
                    'input_data1': {'path': str(input_file1), 'type': 'file'}
                }
                
                if df2 is not None:
                    input_file2 = temp_path / "input_data2.json"
                    df2.to_json(input_file2, orient='records', indent=2)
                    datasets['input_data2'] = {'path': str(input_file2), 'type': 'file'}
                
                # Create operation configuration
                operation_config = self._get_operation_config(operation_type, user_prompt, alias, **kwargs)
                operation_config.update({
                    'litellm_completion_kwargs': {
                        'max_tokens': MAX_TOKENS
                    }
                })
                pipeline_step = self._get_pipeline_step(operation_type)
                
                # Create DocETL pipeline YAML
                pipeline_file = temp_path / "docetl_pipeline.yaml"
                output_file = temp_path / "docetl_output.json"
                api_call_count_file = temp_path / "api_call_count.txt"
                
                pipeline_config = {
                    'datasets': datasets,
                    'default_model': self.model,  # Use configured model
                    'operations': [operation_config],
                    'pipeline': {
                        'steps': [pipeline_step],
                        'output': {'type': 'file', 'path': str(output_file)}
                    }
                }
                
                with open(pipeline_file, 'w') as f:
                    yaml.dump(pipeline_config, f, default_flow_style=False, indent=2)
                
                # Create wrapper script to inject litellm callbacks for API call counting
                wrapper_script = temp_path / "docetl_wrapper.py"
                with open(wrapper_script, 'w') as f:
                    f.write(f'''#!/usr/bin/env python3
import sys
import threading

# CRITICAL: Register callbacks BEFORE importing docetl to ensure they capture all API calls
import litellm

# Global API call counter
_api_call_count = 0
_api_call_lock = threading.Lock()

def _success_callback(kwargs, completion_response, start_time, end_time):
    global _api_call_count
    with _api_call_lock:
        _api_call_count += 1

def _failure_callback(kwargs, completion_response, start_time, end_time):
    global _api_call_count
    with _api_call_lock:
        _api_call_count += 1

# Initialize callback lists if needed
if not hasattr(litellm, 'success_callback') or litellm.success_callback is None:
    litellm.success_callback = []
elif not isinstance(litellm.success_callback, list):
    litellm.success_callback = [litellm.success_callback]

if not hasattr(litellm, 'failure_callback') or litellm.failure_callback is None:
    litellm.failure_callback = []
elif not isinstance(litellm.failure_callback, list):
    litellm.failure_callback = [litellm.failure_callback]

# Register callbacks
litellm.success_callback.append(_success_callback)
litellm.failure_callback.append(_failure_callback)

# NOW import DocETL after callbacks are registered
from docetl.cli import app

# Run DocETL CLI
if __name__ == '__main__':
    try:
        app()
    finally:
        # Write API call count to file
        with open(r"{str(api_call_count_file)}", "w") as f:
            f.write(str(_api_call_count))
''')
                
                # Run wrapper script instead of docetl directly
                cmd = [sys.executable, str(wrapper_script), 'run', str(pipeline_file)]
                env = os.environ.copy()
                
                # Only set OPENAI_API_KEY if we have a valid key (not "dummy")
                if self.api_key and self.api_key != 'dummy':
                    env['OPENAI_API_KEY'] = self.api_key
                
                # Set API base for local VLLM if configured
                if self.api_base:
                    env['HOSTED_VLLM_API_BASE'] = self.api_base
                
                result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, timeout=300, env=env)
                
                # Extract stats from output (including API call count from file)
                # Must do this BEFORE temp_dir is cleaned up
                combined_output = result.stdout + result.stderr
                # Log wrapper debug messages
                for line in result.stderr.split('\n'):
                    if 'WRAPPER' in line or 'CALLBACK' in line:
                        logging.debug(f"Wrapper: {line}")
                # logger.info(f"DocETL output:\n{combined_output}")
                
                # Check if count file exists and log for debugging
                if api_call_count_file.exists():
                    with open(api_call_count_file, 'r') as f:
                        count_content = f.read().strip()
                        logging.debug(f"API call count file exists with content: {count_content}")
                else:
                    logging.debug(f"API call count file does not exist: {api_call_count_file}")
                
                self.last_stats = extract_docetl_stats(combined_output, api_call_count_file=str(api_call_count_file))
                logging.debug(f"DocETL last_stats after extract: api_calls={self.last_stats.api_calls}, cost=${self.last_stats.total_cost}")
                
                if result.returncode != 0:
                    logging.error(f"DocETL {operation_type} command failed with return code {result.returncode}")
                    logging.error(f"STDOUT: {result.stdout}")
                    logging.error(f"STDERR: {result.stderr}")
                    # Raise exception instead of returning input dataframe
                    # This prevents downstream errors when expected columns are missing
                    raise RuntimeError(f"DocETL {operation_type} operation failed: {result.stderr[:500]}")
                
                # Read results
                if output_file.exists():
                    result_df = pd.read_json(output_file)
                    
                    if operation_type == 'cluster':
                        result_df = self._process_cluster_output(result_df)
                    
                    return result_df
                else:
                    logging.error(f"DocETL output file not found: {output_file}")
                    raise RuntimeError(f"DocETL {operation_type} operation failed: output file not found")
                    
        except subprocess.TimeoutExpired:
            logging.error(f"DocETL {operation_type} operation timed out")
            raise RuntimeError(f"DocETL {operation_type} operation timed out after 300 seconds")
        except RuntimeError:
            # Re-raise RuntimeError
            raise
        except Exception as e:
            logging.error(f"Error during DocETL {operation_type}: {e}")
            raise RuntimeError(f"DocETL {operation_type} operation failed: {str(e)}")
    
    def _sem_filter_with_limit(self, df: pd.DataFrame, prompt: str, column_mapping: Dict, limit: int) -> pd.DataFrame:
        """Execute semantic filter with early termination via iterative batching.
        
        This optimization processes the DataFrame in batches and stops once
        enough matching rows are found. This can significantly reduce the
        number of LLM calls for queries with small LIMIT values.
        
        Args:
            df: Input DataFrame (already prepared for DocETL)
            prompt: The semantic filter prompt (already updated for DocETL column names)
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
            return self._run_operation('filter', df, prompt)
        
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
            batch_result = self._run_operation('filter', batch_df, prompt)
            
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

    def sem_where(self, df: pd.DataFrame, user_prompt: str, limit_hint: int = None) -> pd.DataFrame:
        """Semantic filtering using DocETL.
        
        Args:
            df: Input DataFrame to filter
            user_prompt: The semantic filter prompt with column placeholders
            limit_hint: Optional hint for early termination. If provided, the filter
                       will stop processing once enough matching rows are found.
                       This is an optimization hint pushed down from LIMIT clauses.
        """
        df_docetl, column_mapping = self.prepare_dataframe(df)
        updated_prompt = self.update_prompt(user_prompt, column_mapping)
        
        # If limit_hint is provided, use iterative batch processing for early termination
        if limit_hint is not None and limit_hint > 0:
            result_df = self._sem_filter_with_limit(df_docetl, updated_prompt, column_mapping, limit_hint)
        else:
            result_df = self._run_operation('filter', df_docetl, updated_prompt)
        
        if column_mapping:
            result_df = self.restore_dataframe(result_df, column_mapping)
        
        # Ensure columns are preserved even when empty
        if result_df.empty and not df.empty:
            result_df = pd.DataFrame(columns=df.columns)

        logging.info(f"Result of DocETL SEM_WHERE: {result_df.head(10)}")
        return result_df
    
    def sem_where_marker(self, df: pd.DataFrame, user_prompt: str, result_column: str = "_sem_where_result") -> pd.DataFrame:
        """Semantic filter marker using DocETL - adds boolean column instead of filtering.
        
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
        df_docetl, column_mapping = self.prepare_dataframe(df)
        updated_prompt = self.update_prompt(user_prompt, column_mapping)
        
        try:
            # Use sem_filter to get the matching rows
            filtered_df = self._run_operation('filter', df_docetl, updated_prompt)
            
            # Restore column names for comparison
            if column_mapping:
                filtered_df = self.restore_dataframe(filtered_df, column_mapping)
            
            # Get the indices of matching rows
            matching_indices = set(filtered_df.index)
            
            # Create result column: True for matching rows, False otherwise
            result_df = df.copy()
            result_df[result_column] = result_df.index.isin(matching_indices)
            
            logging.info(f"Result of DocETL SEM_WHERE_MARKER: {result_df[result_column].sum()} True out of {len(result_df)} rows")
            return result_df
        except Exception as e:
            logging.error(f"Error during DocETL SEM_WHERE_MARKER: {e}")
            raise RuntimeError(f"DocETL SEM_WHERE_MARKER execution failed: {str(e)}") from e
    
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection using DocETL."""
        # Ensure alias doesn't conflict with existing columns
        unique_alias = self._ensure_unique_alias(df, alias)
        
        df_docetl, column_mapping = self.prepare_dataframe(df)
        updated_prompt = self.update_prompt(user_prompt, column_mapping)

        result_df = self._run_operation('select', df_docetl, updated_prompt, alias=unique_alias)
    
        if not result_df.empty and column_mapping:
            result_df = self.restore_dataframe(result_df, column_mapping)
        
        # Validate result quality
        if unique_alias in result_df.columns:
            null_count = result_df[unique_alias].isna().sum()
            total_count = len(result_df)
            if null_count == total_count and total_count > 0:
                logger.warning(
                    f"DocETL SEM_SELECT column '{unique_alias}' contains only NULL values. "
                    f"Prompt may be unclear: {user_prompt[:80]}..."
                )
            elif null_count > total_count * 0.5:
                logger.warning(
                    f"DocETL SEM_SELECT '{unique_alias}' has {null_count}/{total_count} NULL values."
                )
        
        # Ensure columns are preserved even when empty
        if result_df.empty and not df.empty:
            result_df = pd.DataFrame(columns=df.columns.tolist() + [unique_alias])
        
        logging.info(f"Result of DocETL SEM_SELECT: {result_df.head(10)}")
        return result_df
    
    def sem_join(self, df1: pd.DataFrame, df2: pd.DataFrame, user_prompt: str,
                df1_name: str = "left", df2_name: str = "right",
                limit_hint: Optional[int] = None) -> pd.DataFrame:
        """Semantic join using DocETL.
        
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
        def rename_sem_join_columns(df, left_table, right_table, separator=':'):
            new_columns = []
            for col in df.columns:
                # Replace separator+left and separator+right suffixes
                col = re.sub(rf'^(.*)({re.escape(separator)})left$', rf'{left_table}.\1', col)
                col = re.sub(rf'^(.*)({re.escape(separator)})right$', rf'{right_table}.\1', col)
                new_columns.append(col)
            df.columns = new_columns
            return df
        
        def do_join(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
            """Execute a single join operation."""
            result = self._run_operation('join', left_df, user_prompt, df2=right_df)
            if not result.empty:
                result = rename_sem_join_columns(result, df1_name, df2_name, separator='_')
            return result
        
        # If no limit hint or small tables, do full join
        if limit_hint is None or (len(df1) * len(df2) <= 1000):
            result_df = do_join(df1, df2)
            if result_df.empty:
                logging.error("DocETL join returned an empty DataFrame.")
            else:
                logging.info(f"Result of DocETL SEM_JOIN: {result_df.head(10)}")
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
        
        logging.info(f"Result of DocETL SEM_JOIN: {combined.head(10)}")
        return combined.head(limit_hint) if limit_hint and len(combined) > limit_hint else combined
    
    def sem_group_by(self, df: pd.DataFrame, column: str, number_of_groups: int) -> pd.DataFrame:
        """Semantic grouping using DocETL."""
        if litellm.supports_function_calling(self.model):
            result_df = self._run_operation('cluster', df, '', column=column, number_of_groups=number_of_groups)
            if result_df.empty:
                logging.error("DocETL cluster returned an empty DataFrame.")
                result_df = df
        else:
            logging.error("DocETL requires a LLM that supports function calling for clustering operations.")
            result_df = df
            result_df = self._process_cluster_output(result_df)

        # if result_df.empty:
        #     logging.error("DocETL cluster returned an empty DataFrame.")
        #     result_df = df
        # else:
        #     # Create a mapping of items to their cluster assignments
        #     cluster_mapping = {}
        #     for _, row in result_df.iterrows():
        #         item_value = row[column]
        #         cluster_id = row['cluster_id']
        #         cluster_name = row['cluster_name']
                
        #         if cluster_id not in cluster_mapping:
        #             cluster_mapping[cluster_id] = {
        #                 'name': cluster_name,
        #                 'items': []
        #             }
        #         cluster_mapping[cluster_id]['items'].append(item_value)
            
        #     logging.info(f"Cluster mapping: {cluster_mapping}")

        logging.info(f"Result of DocETL SEM_GROUP_BY: {result_df.head(10)}")
        return result_df
    
    def sem_agg(self, df: pd.DataFrame, user_prompt: str, alias: str,
                group_by_col: Optional[List[str]] = None, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic aggregation using DocETL."""
        if group_by_col and group_by_col[0] in ['cluster_id', 'cluster_name']:
            # Case 1: Aggregation with semantic grouping (SEM_GROUP_BY)
            reduce_key = group_by_col[0]
            
            if column:
                # Focus on specific column with simplified prompt
                docetl_prompt = f"""For group {{{{ reduce_key }}}}, analyze the following {column} values:

{{% for item in inputs %}}
- {{{{ item.{column} }}}}
{{% endfor %}}

{user_prompt}"""
            else:
                # General aggregation with key information only
                docetl_prompt = f"""For group {{{{ reduce_key }}}}, {user_prompt}

Available data:
{{% for item in inputs %}}
Record {{{{ loop.index }}}}: {{{{ item | tojson }}}}
{{% endfor %}}"""
            
            result_df = self._run_operation('reduce', df, docetl_prompt, alias=alias, reduce_key=reduce_key)
            
            # Ensure the result maintains the grouping structure
            if reduce_key not in result_df.columns:
                logging.warning(f"Reduce key '{reduce_key}' not found in result columns: {result_df.columns.tolist()}")
                unique_groups = df[reduce_key].unique()
                if len(result_df) == 1 and len(unique_groups) > 1:
                    logging.warning("Reduce operation collapsed all groups. Reconstructing per-group results...")
                    group_results = []
                    for group_val in unique_groups:
                        group_df = df[df[reduce_key] == group_val]
                        group_result = self._run_operation('reduce', group_df, docetl_prompt, alias=alias, reduce_key=reduce_key)
                        group_result[reduce_key] = group_val
                        group_results.append(group_result)
                    result_df = pd.concat(group_results, ignore_index=True)

        elif group_by_col:
            # Case 2: Aggregation with regular SQL GROUP BY columns
            reduce_key = group_by_col[0]  # Use first GROUP BY column as reduce key
            
            # Validate reduce_key exists in dataframe
            if reduce_key not in df.columns:
                # Check if it's a computed column (like from SUBSTRING_INDEX)
                # Try to extract the actual column name from expressions
                actual_cols = [col for col in df.columns if reduce_key.startswith(col)]
                if not actual_cols:
                    # Cannot find the column - this is likely from SEM_AGG that was called on a dataframe
                    # that doesn't have the expected columns (e.g., SEM_AGG created a new column that isn't in df)
                    logger.error(f"GROUP BY column '{reduce_key}' not found in DataFrame columns: {df.columns.tolist()}")
                    raise ValueError(f"GROUP BY column '{reduce_key}' not found in DataFrame columns: {df.columns.tolist()}")
            
            if column:
                # Focus on specific column for aggregation
                # Use reduce_key.column_name to access the reduce key value
                docetl_prompt = f"""For {reduce_key} = "{{{{ reduce_key.{reduce_key} }}}}", analyze the {column} values:

{{% for item in inputs %}}
- {{{{ item.{column} }}}}
{{% endfor %}}

{user_prompt}"""
            else:
                # General aggregation
                # Use reduce_key.column_name to access the reduce key value
                docetl_prompt = f"""For {reduce_key} = "{{{{ reduce_key.{reduce_key} }}}}", {user_prompt}

Available data:
{{% for item in inputs %}}
{{{{ item | tojson }}}}
{{% endfor %}}"""
            
            result_df = self._run_operation('reduce', df, docetl_prompt, alias=alias, reduce_key=reduce_key)

        elif column:
            # Case 3: Column-specific aggregation without grouping
            if column not in df.columns:
                raise ValueError(f"Aggregation column '{column}' not found in DataFrame columns: {df.columns.tolist()}")
            
            # Create a single group for all values of the specified column
            df_grouped = df.copy()
            df_grouped['_agg_group'] = 'all_values'
            
            docetl_prompt = f"""Analyze all the {column} values:

{{% for item in inputs %}}
- {{{{ item.{column} }}}}
{{% endfor %}}

{user_prompt}"""
            
            result_df = self._run_operation('reduce', df_grouped, docetl_prompt, alias=alias, reduce_key='_agg_group')
            if '_agg_group' in result_df.columns:
                result_df = result_df.drop(columns=['_agg_group'])

        else:
            # Case 4: Global aggregation across all records
            df_with_dummy = df.copy()
            df_with_dummy['_global_group'] = 'all'
            
            # Simple prompt for global aggregation
            docetl_prompt = f"""{user_prompt}

Available data: {{{{ inputs | length }}}} records total.
{{% for item in inputs %}}
Record {{{{ loop.index }}}}: {{{{ item | tojson }}}}
{{% endfor %}}"""
            
            result_df = self._run_operation('reduce', df_with_dummy, docetl_prompt, alias=alias, reduce_key='_global_group')
            if '_global_group' in result_df.columns:
                result_df = result_df.drop(columns=['_global_group'])

        logging.info(f"Result of DocETL SEM_AGG: {result_df.head(10)}")
        return result_df
    
    def sem_distinct(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Semantic deduplication using DocETL."""
        result_df = self._run_operation('resolve', df, '', column=column, threshold=DOCETL_DEFAULT_RESOLVE_THRESHOLD)

        # logging.info(f"Result after resolve operation: {len(result_df)} rows")
        # logging.info(f"Result of DocETL Resolve: {result_df.head(10)}")
        # # remove _kv_pairs_preresolve_resolve_distinct column
        # if '_kv_pairs_preresolve_resolve_distinct' in result_df.columns:
        #     result_df = result_df.drop(columns=['_kv_pairs_preresolve_resolve_distinct'])
        #     # logging.info(f"Result of DocETL Resolve after dropping internal column: {result_df.head(10)}")
        
        if len(result_df) > 0:
            # Keep only one row per unique value in the resolved column
            result_df = result_df.drop_duplicates(subset=[column], keep='first')
            # logging.info(f"Final deduplicated result: {len(result_df)} rows")
            # logging.info(f"Final result after deduplication: {result_df.head(10)}")
        
        # in '_kv_pairs_preresolve_resolve_distinct' column, there are key-value pairs of original columns
        # restore original columns from there for the case when original values are needed
        # then replace the column with the resolved value
        if '_kv_pairs_preresolve_resolve_distinct' in result_df.columns:
            def restore_original_columns(row):
                kv_pairs = row['_kv_pairs_preresolve_resolve_distinct']
                if isinstance(kv_pairs, dict):
                    for key, value in kv_pairs.items():
                        if key == column:
                            row[key] = value
                return row
            
            result_df = result_df.apply(restore_original_columns, axis=1)
            result_df = result_df.drop(columns=['_kv_pairs_preresolve_resolve_distinct'])
            # logging.info(f"Result after restoring original columns: {result_df.head(10)}")

        logging.info(f"Result of DocETL SEM_DISTINCT: {result_df.head(10)}")
        return result_df
    
    def sem_order_by(self, df: pd.DataFrame, user_prompt: str, column: Optional[str] = None) -> pd.DataFrame:
        """Semantic ordering using DocETL."""
        result_df = self._run_operation('rank', df, user_prompt, 
                                        df_columns=df.columns.tolist(),
                                        direction='desc', input_keys=[column] if column else None, call_budget=10, initial_ordering_method='likert')
        
        if result_df.empty:
            logging.error("DocETL rank returned an empty DataFrame.")
            result_df = df
        
        logging.info(f"Result of DocETL SEM_ORDER_BY: {result_df.head(10)}")
        return result_df

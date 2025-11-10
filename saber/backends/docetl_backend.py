"""
DocETL backend implementation for SABER semantic operations.
"""
import os
import re
import subprocess
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import logging
import pandas as pd

from .base_backend import BaseBackend
from ..config import DOCETL_DEFAULT_MODEL, DOCETL_DEFAULT_EMBEDDING_MODEL, DOCETL_DEFAULT_RESOLVE_THRESHOLD
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
        """Update prompts to use underscore column names for DocETL."""
        updated_prompt = prompt
        
        # Replace dot notation with underscore notation in DocETL templates
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
                    }
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
                    }
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
                    }
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
                    }
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
                    logging.info("No cluster column found, adding default cluster_id")
                    df['cluster_id'] = 'cluster_001'
                    return df
            
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
                pipeline_step = self._get_pipeline_step(operation_type)
                
                # Create DocETL pipeline YAML
                pipeline_file = temp_path / "docetl_pipeline.yaml"
                output_file = temp_path / "docetl_output.json"
                
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
                
                # Run DocETL via subprocess with API key in environment
                cmd = ['docetl', 'run', str(pipeline_file)]
                env = os.environ.copy()
                
                # Only set OPENAI_API_KEY if we have a valid key (not "dummy")
                if self.api_key and self.api_key != 'dummy':
                    env['OPENAI_API_KEY'] = self.api_key
                
                # Set API base for local VLLM if configured
                if self.api_base:
                    env['HOSTED_VLLM_API_BASE'] = self.api_base
                
                result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, timeout=300, env=env)
                
                # Extract stats from output
                combined_output = result.stdout + result.stderr
                # logger.info(f"DocETL output:\n{combined_output}")
                self.last_stats = extract_docetl_stats(combined_output)
                
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
    
    def sem_where(self, df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
        """Semantic filtering using DocETL."""
        df_docetl, column_mapping = self.prepare_dataframe(df)
        updated_prompt = self.update_prompt(user_prompt, column_mapping)
        
        result_df = self._run_operation('filter', df_docetl, updated_prompt)
        
        if not result_df.empty and column_mapping:
            result_df = self.restore_dataframe(result_df, column_mapping)

        logging.info(f"Result of DocETL SEM_WHERE: {result_df.head(10)}")
        return result_df
    
    def sem_select(self, df: pd.DataFrame, user_prompt: str, alias: str) -> pd.DataFrame:
        """Semantic selection using DocETL."""
        df_docetl, column_mapping = self.prepare_dataframe(df)
        updated_prompt = self.update_prompt(user_prompt, column_mapping)

        result_df = self._run_operation('select', df_docetl, updated_prompt, alias=alias)
    
        if not result_df.empty and column_mapping:
            result_df = self.restore_dataframe(result_df, column_mapping)
        
        logging.info(f"Result of DocETL SEM_SELECT: {result_df.head(10)}")
        return result_df
    
    def sem_join(self, df1: pd.DataFrame, df2: pd.DataFrame, user_prompt: str,
                df1_name: str = "left", df2_name: str = "right") -> pd.DataFrame:
        """Semantic join using DocETL."""
        def rename_sem_join_columns(df, left_table, right_table, separator=':'):
            new_columns = []
            for col in df.columns:
                # Replace separator+left and separator+right suffixes
                col = re.sub(rf'^(.*)({re.escape(separator)})left$', rf'{left_table}.\1', col)
                col = re.sub(rf'^(.*)({re.escape(separator)})right$', rf'{right_table}.\1', col)
                new_columns.append(col)
            df.columns = new_columns
            return df
        
        result_df = self._run_operation('join', df1, user_prompt, df2=df2)
        if not result_df.empty:
            result_df = rename_sem_join_columns(result_df, df1_name, df2_name, separator='_')
        else:
            logging.error("DocETL join returned an empty DataFrame.")

        logging.info(f"Result of DocETL SEM_JOIN: {result_df.head(10)}")
        return result_df
    
    def sem_group_by(self, df: pd.DataFrame, column: str, number_of_groups: int) -> pd.DataFrame:
        """Semantic grouping using DocETL."""
        result_df = self._run_operation('cluster', df, '', column=column, number_of_groups=number_of_groups)

        if result_df.empty:
            logging.error("DocETL cluster returned an empty DataFrame.")
            result_df = df
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
            
            if reduce_key not in df.columns:
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

"""
Query rewriter for backend-specific prompt generation.
"""
import re
import json
from typing import Dict, Set, List, Tuple
import logging

import litellm

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Rewrites backend-free queries to backend-specific ones."""
    
    def __init__(self, model_name: str, api_base: str = None, api_key: str = None):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = 3
    
    def get_backend_system_prompt(self, backend: str, aliases: Dict[str, str], operation: str = None) -> str:
        """Get backend-specific system prompt for query rewriting."""
        alias_info = "Table aliases found in the query: " + json.dumps(aliases) + "\n" if aliases else ""
        operation_info = f"Target semantic operation: {operation}\n" if operation else ""
        
        if backend == 'lotus':
            return f"""You are an expert in rewriting semantic SQL queries for the LOTUS backend system. LOTUS is a semantic operator library that processes natural language queries over semi-structured data using large language models.

{alias_info}{operation_info}
For LOTUS backend, you must:
1. **Use ONLY column names that exist in the schema** - do not invent or infer new column names
2. **Explicitly reference column names in curly braces** within the prompt text
3. **Use table aliases with dot notation when aliases exist** (e.g., `{{r.cuisine_type}}`, `{{c.experience}}`)
4. **Use direct column names** when no aliases exist (e.g., `{{description}}`, `{{cuisine_type}}`)
5. **Pay attention to the SQL context** - if the query has aliases like `FROM restaurants AS r JOIN chefs AS c`, then use `{{r.column_name}}` and `{{c.column_name}}`
6. **Keep prompts concise and focused** on the semantic matching task
7. **Do NOT add operation names or prefixes** to your response
8. **NEVER use single quotes (') or double quotes (") anywhere in your response**

Operation-specific guidelines:
- **SEM_WHERE/SELECT**: Use `{{column}}` or `{{alias.column}}` format with actual column names from schema
- **SEM_ORDER_BY**: Use `{{column}}` format to indicate which column to compare
- **SEM_AGG**: Use `{{column}}` format when aggregating over a specific column
- **SEM_JOIN**: Use `{{column:left}}` and `{{column:right}}` to reference columns from different tables

Examples of good LOTUS prompts (with and without aliases):
- SEM_WHERE (no alias): {{product_name}} is related to electronics or technology
- SEM_WHERE (with alias): {{r.description}} focuses on authentic or traditional cuisine
- SEM_SELECT (no alias): What is the most famous trademark for {{name}}? Write only trademark.
- SEM_SELECT (with alias): Summarize the {{c.experience}} focusing on chef specialty in one sentence.
- SEM_ORDER_BY (no alias): Which {{name}} is the most delicious in winter?
- SEM_ORDER_BY (with alias): Which {{r.rating}} indicates highest quality?
- SEM_AGG (no alias): Analyze the status of {{note}} and summarize it in one sentence.
- SEM_AGG (with alias): Summarize all {{r.description}} and {{c.experience}} in one sentence.
- SEM_JOIN: {{name:left}} is used in {{name:right}}

**CRITICAL**: Your response must contain ONLY the rewritten prompt text. Do not include any explanatory text, operation names, markdown, quotation marks, or any other formatting. The response must be plain text without any quotes whatsoever."""

        elif backend == 'docetl':
            return f"""You are an expert in rewriting semantic SQL queries for the DocETL backend system. DocETL is a document processing pipeline that uses large language models to analyze and transform unstructured data with template-based prompts.

{alias_info}{operation_info}
For DocETL backend, you must:
1. **Use Jinja2 template syntax** with `{{{{ input.column_name }}}}` format
2. **When table aliases exist in the SQL query**, use dot notation: `{{{{ input.alias.column_name }}}}` (e.g., `{{{{ input.r.description }}}}`, `{{{{ input.c.experience }}}}`)
3. **When no aliases exist**, use direct column names: `{{{{ input.column_name }}}}` (e.g., `{{{{ input.description }}}}`, `{{{{ input.cuisine_type }}}}`)
4. **Pay attention to the SQL context** - if the query has `FROM restaurants AS r JOIN chefs AS c`, then use `{{{{ input.r.column_name }}}}` and `{{{{ input.c.column_name }}}}`
5. **Provide clear context and instructions** for the LLM to follow
6. **Structure prompts with explicit formatting** and expected output guidance
7. **Include relevant data context** to help the LLM make informed decisions
8. **NEVER use single quotes (') or double quotes (") anywhere in your response**

Operation-specific guidelines:
- **SEM_WHERE**: Use `{{{{ input.column }}}}` format with clear True/False instructions
- **SEM_SELECT**: Use `{{{{ input.column }}}}` format with clear extraction instructions
- **SEM_ORDER_BY**: Describe comparison criteria (column arg is separate)
- **SEM_AGG**: For column-specific agg, column arg is separate; prompt describes aggregation
- **SEM_JOIN**: Use `{{{{ left.column }}}}` and `{{{{ right.column }}}}` for comparing rows from two tables

Examples of good DocETL prompts (with and without aliases):
- SEM_WHERE (no alias): Product Description: {{{{ input.description }}}}\n\nAnalyze if this product targets professional users and return True or False.
- SEM_WHERE (with alias): Restaurant Description: {{{{ input.r.description }}}}\n\nReturn True if the restaurant focuses on authentic or traditional cuisine, otherwise return False.
- SEM_SELECT (no alias): {{{{ input.name }}}}\n\nWrite the most famous trademark.
- SEM_SELECT (with alias): Chef Experience: {{{{ input.c.experience }}}}\n\nSummarize the chef specialty or unique cooking technique mentioned.
- SEM_ORDER_BY (no alias): Order products based on how delicious they are in winter?
- SEM_ORDER_BY (with alias): Order restaurants based on their rating quality?
- SEM_AGG (no alias): Summarize all product names in one sentence.
- SEM_AGG (with alias): Summarize the restaurant descriptions and chef experiences in one sentence.
- SEM_JOIN: Compare the following product and dessert:\nProduct Name: {{{{ left.name }}}}\nDessert Name: {{{{ right.name }}}}\n\nIs this product a good match for the dessert? Respond with True if it is a good match or False if it is not a suitable match.

**IMPORTANT**: Your response must contain ONLY the rewritten prompt text. Do not include any explanatory text, markdown, quotation marks, or any other formatting. The response must be plain text without any quotes whatsoever."""

        else:
            # Fallback generic prompt
            return f"""You are an expert in rewriting semantic SQL queries. Your task is to convert a backend-free semantic query part into a backend-specific one. The target backend is '{backend}'.

{alias_info}{operation_info}
Based on the user's prompt and the provided schema, rewrite the prompt to be compatible with the specified backend. Infer the most relevant column(s) from the schema and incorporate them appropriately.

**IMPORTANT**: Your response must contain ONLY the rewritten prompt text. Do not include any explanatory text, markdown, quotation marks, or any other formatting. The response must be plain text without any quotes whatsoever."""

    def rewrite_prompt(self, user_prompt: str, column_context: str, aliases: Dict[str, str], backend: str, operation: str = None) -> str:
        """Rewrite a backend-free prompt to a backend-specific one using an LLM."""
        if backend == 'palimpzest':
            # Palimpzest uses more natural language, so minimal rewriting needed
            return user_prompt

        # Build a set of valid column identifiers from the provided column_context.
        valid_columns = self._parse_column_context(column_context)

        # Try rewriting up to max_retries times, verify each rewrite programmatically
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            # Get backend-specific system prompt
            system_prompt = self.get_backend_system_prompt(backend, aliases, operation)

            # Build completion parameters
            completion_params = {
                'model': self.model_name,
                'messages': [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Prompt: '{user_prompt}'\n\n{column_context}"}
                ]
            }
            
            # Add optional parameters
            if self.api_base:
                completion_params['api_base'] = self.api_base
            if self.api_key:
                completion_params['api_key'] = self.api_key
            
            completion = litellm.completion(**completion_params)
            rewritten_prompt = completion.choices[0].message.content.strip()

            # Remove any stray quotes the LLM might add
            rewritten_prompt = self._remove_quotes(rewritten_prompt)

            # Verify rewritten prompt syntactically against known columns for target backend
            try:
                ok = self._verify_prompt(backend, rewritten_prompt, valid_columns, operation)
            except Exception as e:
                ok = False
                last_error = e

            if ok:
                return rewritten_prompt

            # Not ok -> record and retry (loop will try again)
            last_error = last_error or Exception("Verification failed")

        # If we get here, all retries failed. Return the last rewrite (best effort) and log the issue.
        logger.warning(f"QueryRewriter: failed to verify rewritten prompt for backend='{backend}' after {self.max_retries} attempts. Last error: {last_error}")
        return rewritten_prompt

    def _parse_column_context(self, column_context: str) -> Set[str]:
        """Extract a set of available column identifiers from the column context string.

        This returns both dotted (alias.column) and plain column names where possible.
        """
        cols: Set[str] = set()
        if not column_context:
            return cols

        # Find alias.column patterns first
        for m in re.finditer(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", column_context):
            alias = m.group(1)
            col = m.group(2)
            cols.add(f"{alias}.{col}")
            cols.add(col)

        # Also pick up standalone column-like tokens (best-effort)
        for m in re.finditer(r"\b([a-zA-Z_][\w]*)\b", column_context):
            token = m.group(1)
            cols.add(token)

        return cols

    def _verify_prompt(self, backend: str, prompt: str, valid_columns: Set[str], operation: str = None) -> bool:
        """Verify the rewritten prompt contains backend-appropriate column references.

        Verification rules vary by backend and operation type:
        - LOTUS: Different operations have different column reference requirements
        - DocETL: Different operations have different template patterns
        - Other backends: Permissive validation
        """
        if backend == 'lotus':
            return self._verify_lotus_prompt(prompt, valid_columns, operation)
        elif backend == 'docetl':
            return self._verify_docetl_prompt(prompt, valid_columns, operation)
        else:
            # Fallback: if we don't have specific rules for the backend, accept the prompt
            return True

    def _verify_lotus_prompt(self, prompt: str, valid_columns: Set[str], operation: str = None) -> bool:
        """Verify LOTUS prompt based on operation type.
        
        Operation-specific rules:
        - SEM_WHERE/SEM_SELECT/SEM_AGG/SEM_ORDER_BY: Must have at least one {column} reference
        - SEM_JOIN: Must have {column:left} and {column:right} references
        - SEM_GROUP_BY/SEM_DISTINCT: Column is passed separately, prompt may not need braces
        """
        # Find all {expr} occurrences (standard format)
        standard_refs = re.findall(r"\{([^}:]+)\}", prompt)
        # Find all {expr:side} occurrences (join format)
        join_refs = re.findall(r"\{([^}:]+):(left|right)\}", prompt)
        
        if operation in ['SEM_WHERE', 'SEM_SELECT', 'SEM_AGG', 'SEM_ORDER_BY']:
            # These operations require explicit column references
            if not standard_refs:
                return False
            # Verify all referenced columns exist
            for expr in standard_refs:
                expr = expr.strip()
                if not self._is_valid_column_ref(expr, valid_columns):
                    return False
            return True
        
        elif operation == 'SEM_JOIN':
            # JOIN requires special {column:left} and {column:right} format
            if not join_refs:
                return False
            has_left = any(side == 'left' for _, side in join_refs)
            has_right = any(side == 'right' for _, side in join_refs)
            if not (has_left and has_right):
                return False
            # Verify all join columns exist
            for expr, _ in join_refs:
                expr = expr.strip()
                if not self._is_valid_column_ref(expr, valid_columns):
                    return False
            return True
        
        elif operation in ['SEM_GROUP_BY', 'SEM_DISTINCT']:
            # These operations may not require column refs in prompt (column passed separately)
            # If there are refs, they should be valid
            if standard_refs:
                for expr in standard_refs:
                    expr = expr.strip()
                    if not self._is_valid_column_ref(expr, valid_columns):
                        return False
            return True
        
        else:
            # Unknown operation: require at least some valid column reference if braces are present
            if standard_refs:
                for expr in standard_refs:
                    expr = expr.strip()
                    if not self._is_valid_column_ref(expr, valid_columns):
                        return False
                return True
            # No column refs but that might be okay for unknown operations
            return True

    def _verify_docetl_prompt(self, prompt: str, valid_columns: Set[str], operation: str = None) -> bool:
        """Verify DocETL prompt based on operation type.
        
        Operation-specific rules:
        - SEM_WHERE: Must have {{ input.column }} or {{ input.alias.column }} with True/False guidance
        - SEM_SELECT: Must have {{ input.column }} or {{ input.alias.column }} with extraction guidance
        - SEM_ORDER_BY: May describe comparison without explicit column refs (column passed separately)
        - SEM_AGG: May describe aggregation without explicit column refs (column passed separately)
        - SEM_JOIN: Must have {{ left.column }} and {{ right.column }} references
        - SEM_GROUP_BY/SEM_DISTINCT: Column is passed separately, prompt describes grouping logic
        """
        # Match {{ input.expr }} (can be input.column or input.alias.column)
        # Pattern captures everything after 'input.' until closing braces
        input_refs = re.findall(r"\{\{\s*input\.([^}\s]+?)\s*\}\}", prompt)
        # Match {{ left.expr }} and {{ right.expr }} (join format)
        left_refs = re.findall(r"\{\{\s*left\.([^}\s]+?)\s*\}\}", prompt)
        right_refs = re.findall(r"\{\{\s*right\.([^}\s]+?)\s*\}\}", prompt)
        
        if operation in ['SEM_WHERE', 'SEM_SELECT']:
            # These operations require explicit {{ input.column }} references
            if not input_refs:
                return False
            # Verify all referenced columns exist
            for expr in input_refs:
                expr = expr.strip()
                if not self._is_valid_column_ref(expr, valid_columns):
                    return False
            return True
        
        elif operation in ['SEM_ORDER_BY', 'SEM_AGG', 'SEM_GROUP_BY', 'SEM_DISTINCT']:
            # These operations may not require column refs in prompt (column passed separately)
            # But if there are refs, they should be valid
            if input_refs:
                for expr in input_refs:
                    expr = expr.strip()
                    if not self._is_valid_column_ref(expr, valid_columns):
                        return False
            # Accept prompts without column refs for these operations
            return True
        
        elif operation == 'SEM_JOIN':
            # JOIN requires {{ left.column }} and {{ right.column }} format
            if not (left_refs and right_refs):
                return False
            # Verify all join columns exist
            for expr in left_refs + right_refs:
                expr = expr.strip()
                if not self._is_valid_column_ref(expr, valid_columns):
                    return False
            return True
        
        else:
            # Unknown operation: if there are input refs, they should be valid
            if input_refs:
                for expr in input_refs:
                    expr = expr.strip()
                    if not self._is_valid_column_ref(expr, valid_columns):
                        return False
                return True
            # No column refs might be okay for unknown operations
            return True

    def _is_valid_column_ref(self, expr: str, valid_columns: Set[str]) -> bool:
        """Check if a column reference expression is valid.
        
        Valid forms:
        - exact match: 'alias.column' or 'column'
        - partial match: if expr is 'alias.column', 'column' should exist
        """
        # Direct match
        if expr in valid_columns:
            return True
        # If expr contains a dot, check if the column part exists
        if '.' in expr:
            col_part = expr.split('.')[-1]
            if col_part in valid_columns:
                return True
            # Also check the full dotted name
            if expr in valid_columns:
                return True
        return False
    
    def _remove_quotes(self, text: str) -> str:
        """Remove all types of quotes from text."""
        # Remove all single and double quotes
        text = text.replace("'", "")
        text = text.replace('"', "")
        
        # Remove smart/curly quotes
        text = text.replace("‘", "")  # Left single quotation mark (U+2018)
        text = text.replace("’", "")  # Right single quotation mark (U+2019)
        text = text.replace("“", "")  # Left double quotation mark (U+201C)
        text = text.replace("”", "")  # Right double quotation mark (U+201D)
        
        # Remove backticks
        text = text.replace("`", "")
        
        return text

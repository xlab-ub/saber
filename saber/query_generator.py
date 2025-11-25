"""
SABER Query Generator - Generate SABER SQL from natural language.
"""
from typing import Dict, List, Optional, Set
import logging
import re
# from textwrap import dedent

import litellm

logger = logging.getLogger(__name__)


class SABERQueryGenerator:
    """Generate SABER SQL queries from natural language questions."""
    
    @staticmethod
    def _clean_column_name(col_name: str) -> str:
        """Remove backticks, quotes, and extra spaces from column names."""
        return col_name.strip().replace('`', '').replace('"', '').replace("'", '').strip()
    
    @staticmethod
    def _extract_column_references(text: str) -> List[str]:
        """Extract column references from {column} patterns in text."""
        return [match.strip() for match in re.findall(r'\{([^}]+)\}', text)]
    
    @staticmethod
    def _ensure_column_braces(prompt: str, available_columns: Set[str]) -> str:
        """Ensure column references in prompts use curly braces {column}."""
        # If prompt has no column references, warn
        if '{' not in prompt and '}' not in prompt:
            logger.warning(f"Semantic function prompt missing column references ({{column}}): {prompt[:80]}...")
        return prompt
    
    DEFAULT_EXAMPLES = {
        'lotus': [
            {
                "schema": """
Table: customer_reviews
Columns: review_id (int64), product_name (object), rating (int64), review_text (object)
Sample rows (5):
  1, 'Laptop Pro 15', 5, 'Excellent build quality and fast performance for coding'
  2, 'Wireless Mouse', 4, 'Comfortable grip but battery life could be better'
  3, 'USB-C Hub', 3, 'Works fine but gets warm during extended use'
  4, 'Mechanical Keyboard', 5, 'Solid construction and very durable, worth the price'
  5, 'Monitor 27inch', 2, 'Poor quality, screen flickering after one week'
""",
                "question": "Which reviews mention build quality or durability?",
                "saber_sql": """
SELECT product_name, rating, review_text
FROM customer_reviews
WHERE SEM_WHERE('{review_text} discusses build quality or durability', 'lotus')
"""
            },
            {
                "schema": """
Table: job_postings
Columns: job_id (int64), company (object), title (object), location (object), description (object)
Sample rows (5):
  101, 'TechCorp', 'Senior Engineer', 'San Francisco, CA', 'Required: 5+ years Python, AWS experience'
  102, 'DataSys', 'ML Engineer', 'New York, NY', 'Required: 3+ years machine learning, TensorFlow'
  103, 'CloudNet', 'DevOps Lead', 'Austin, TX', 'Required: 7+ years Kubernetes, CI/CD pipelines'
  104, 'StartupXYZ', 'Backend Dev', 'Remote', 'Required: 2+ years Node.js, MongoDB experience'
  105, 'FinanceHub', 'Data Analyst', 'Boston, MA', 'Required: 4+ years SQL, Tableau, statistics'
""",
                "question": "How many postings require each experience level?",
                "saber_sql": """
SELECT SEM_SELECT('Extract years of experience from {description} as a number without explanation', 'lotus') AS years_exp,
       COUNT(*) AS posting_count
FROM job_postings
GROUP BY years_exp
"""
            },
            {
                "schema": """
Table: email_campaigns
Columns: campaign_id (int64), subject (object), content (object), open_rate (float64), click_rate (float64)
Sample rows (5):
  1, 'New Features Release', 'We are excited to announce new AI-powered analytics...', 0.45, 0.12
  2, 'Summer Sale 50% Off', 'Limited time offer on all premium plans...', 0.62, 0.28
  3, 'Product Tutorial Series', 'Learn advanced data visualization techniques...', 0.38, 0.09
  4, 'Customer Success Story', 'How TechCorp improved efficiency with our platform...', 0.51, 0.15
  5, 'Security Update Notice', 'Important security patches and best practices...', 0.71, 0.22
""",
                "question": "What are the top 3 most engaging campaigns about product features or updates?",
                "saber_sql": """
SELECT campaign_id,
       subject,
       SEM_SELECT('Extract main topic from {content} (e.g., features, tutorial, sale, security) without explanation', 'lotus') AS topic,
       open_rate,
       click_rate
FROM email_campaigns
WHERE SEM_WHERE('{content} discusses product features, updates, or new capabilities', 'lotus')
ORDER BY click_rate DESC
LIMIT 3
"""
            }
        ],
        'docetl': [
            {
                "schema": """
Table: customer_reviews
Columns: review_id (int64), product_name (object), rating (int64), review_text (object)
Sample rows (5):
  1, 'Laptop Pro 15', 5, 'Excellent build quality and fast performance for coding'
  2, 'Wireless Mouse', 4, 'Comfortable grip but battery life could be better'
  3, 'USB-C Hub', 3, 'Works fine but gets warm during extended use'
  4, 'Mechanical Keyboard', 5, 'Solid construction and very durable, worth the price'
  5, 'Monitor 27inch', 2, 'Poor quality, screen flickering after one week'
""",
                "question": "Which reviews mention build quality or durability?",
                "saber_sql": """
SELECT product_name, rating, review_text
FROM customer_reviews
WHERE SEM_WHERE('Review text: {{ input.review_text }}

Return true if the review discusses build quality or durability, otherwise false.', 'docetl')
"""
            },
            {
                "schema": """
Table: job_postings
Columns: job_id (int64), company (object), title (object), location (object), description (object)
Sample rows (5):
  101, 'TechCorp', 'Senior Engineer', 'San Francisco, CA', 'Required: 5+ years Python, AWS experience'
  102, 'DataSys', 'ML Engineer', 'New York, NY', 'Required: 3+ years machine learning, TensorFlow'
  103, 'CloudNet', 'DevOps Lead', 'Austin, TX', 'Required: 7+ years Kubernetes, CI/CD pipelines'
  104, 'StartupXYZ', 'Backend Dev', 'Remote', 'Required: 2+ years Node.js, MongoDB experience'
  105, 'FinanceHub', 'Data Analyst', 'Boston, MA', 'Required: 4+ years SQL, Tableau, statistics'
""",
                "question": "How many postings require each experience level?",
                "saber_sql": """
SELECT SEM_SELECT('Description: {{ input.description }}

Extract the years of experience required (return as a single number) without explanation.', 'docetl') AS years_exp,
       COUNT(*) AS posting_count
FROM job_postings
GROUP BY years_exp
"""
            },
            {
                "schema": """
Table: email_campaigns
Columns: campaign_id (int64), subject (object), content (object), open_rate (float64), click_rate (float64)
Sample rows (5):
  1, 'New Features Release', 'We are excited to announce new AI-powered analytics...', 0.45, 0.12
  2, 'Summer Sale 50% Off', 'Limited time offer on all premium plans...', 0.62, 0.28
  3, 'Product Tutorial Series', 'Learn advanced data visualization techniques...', 0.38, 0.09
  4, 'Customer Success Story', 'How TechCorp improved efficiency with our platform...', 0.51, 0.15
  5, 'Security Update Notice', 'Important security patches and best practices...', 0.71, 0.22
""",
                "question": "What are the top 3 most engaging campaigns about product features or updates?",
                "saber_sql": """
SELECT campaign_id,
       subject,
       SEM_SELECT('Content: {{ input.content }}

Extract main topic (features, tutorial, sale, or security)' without explanation', 'docetl') AS topic,
       open_rate,
       click_rate
FROM email_campaigns
WHERE SEM_WHERE('Content: {{ input.content }}

Return true if discusses product features, updates, or new capabilities, otherwise false.', 'docetl')
ORDER BY click_rate DESC
LIMIT 3
"""
            }
        ],
        'palimpzest': [
            {
                "schema": """
Table: customer_reviews
Columns: review_id (int64), product_name (object), rating (int64), review_text (object)
Sample rows (5):
  1, 'Laptop Pro 15', 5, 'Excellent build quality and fast performance for coding'
  2, 'Wireless Mouse', 4, 'Comfortable grip but battery life could be better'
  3, 'USB-C Hub', 3, 'Works fine but gets warm during extended use'
  4, 'Mechanical Keyboard', 5, 'Solid construction and very durable, worth the price'
  5, 'Monitor 27inch', 2, 'Poor quality, screen flickering after one week'
""",
                "question": "Which reviews mention build quality or durability?",
                "saber_sql": """
SELECT product_name, rating, review_text
FROM customer_reviews
WHERE SEM_WHERE('The review mentions build quality or durability', 'palimpzest')
"""
            },
            {
                "schema": """
Table: job_postings
Columns: job_id (int64), company (object), title (object), location (object), description (object)
Sample rows (5):
  101, 'TechCorp', 'Senior Engineer', 'San Francisco, CA', 'Required: 5+ years Python, AWS experience'
  102, 'DataSys', 'ML Engineer', 'New York, NY', 'Required: 3+ years machine learning, TensorFlow'
  103, 'CloudNet', 'DevOps Lead', 'Austin, TX', 'Required: 7+ years Kubernetes, CI/CD pipelines'
  104, 'StartupXYZ', 'Backend Dev', 'Remote', 'Required: 2+ years Node.js, MongoDB experience'
  105, 'FinanceHub', 'Data Analyst', 'Boston, MA', 'Required: 4+ years SQL, Tableau, statistics'
""",
                "question": "How many postings require each experience level?",
                "saber_sql": """
SELECT SEM_SELECT('Extract the required years of experience as a number without explanation', 'palimpzest') AS years_exp,
       COUNT(*) AS posting_count
FROM job_postings
GROUP BY years_exp
"""
            },
            {
                "schema": """
Table: email_campaigns
Columns: campaign_id (int64), subject (object), content (object), open_rate (float64), click_rate (float64)
Sample rows (5):
  1, 'New Features Release', 'We are excited to announce new AI-powered analytics...', 0.45, 0.12
  2, 'Summer Sale 50% Off', 'Limited time offer on all premium plans...', 0.62, 0.28
  3, 'Product Tutorial Series', 'Learn advanced data visualization techniques...', 0.38, 0.09
  4, 'Customer Success Story', 'How TechCorp improved efficiency with our platform...', 0.51, 0.15
  5, 'Security Update Notice', 'Important security patches and best practices...', 0.71, 0.22
""",
                "question": "What are the top 3 most engaging campaigns about product features or updates?",
                "saber_sql": """
SELECT campaign_id,
       subject,
       SEM_SELECT('Extract main topic: features, tutorial, sale, or security without explanation', 'palimpzest') AS topic,
       open_rate,
       click_rate
FROM email_campaigns
WHERE SEM_WHERE('This campaign discusses product features, updates, or new capabilities', 'palimpzest')
ORDER BY click_rate DESC
LIMIT 3
"""
            }
        ],
        'unified': [
            {
                "schema": """
Table: customer_reviews
Columns: review_id (int64), product_name (object), rating (int64), review_text (object)
Sample rows (5):
  1, 'Laptop Pro 15', 5, 'Excellent build quality and fast performance for coding'
  2, 'Wireless Mouse', 4, 'Comfortable grip but battery life could be better'
  3, 'USB-C Hub', 3, 'Works fine but gets warm during extended use'
  4, 'Mechanical Keyboard', 5, 'Solid construction and very durable, worth the price'
  5, 'Monitor 27inch', 2, 'Poor quality, screen flickering after one week'
""",
                "question": "Which reviews mention build quality or durability?",
                "saber_sql": """
SELECT product_name, rating, review_text
FROM customer_reviews
WHERE SEM_WHERE('The review discusses build quality or durability')
"""
            },
            {
                "schema": """
Table: job_postings
Columns: job_id (int64), company (object), title (object), location (object), description (object)
Sample rows (5):
  101, 'TechCorp', 'Senior Engineer', 'San Francisco, CA', 'Required: 5+ years Python, AWS experience'
  102, 'DataSys', 'ML Engineer', 'New York, NY', 'Required: 3+ years machine learning, TensorFlow'
  103, 'CloudNet', 'DevOps Lead', 'Austin, TX', 'Required: 7+ years Kubernetes, CI/CD pipelines'
  104, 'StartupXYZ', 'Backend Dev', 'Remote', 'Required: 2+ years Node.js, MongoDB experience'
  105, 'FinanceHub', 'Data Analyst', 'Boston, MA', 'Required: 4+ years SQL, Tableau, statistics'
""",
                "question": "How many postings require each experience level?",
                "saber_sql": """
SELECT SEM_SELECT('Extract years of experience required as a number without explanation') AS years_exp,
       COUNT(*) AS posting_count
FROM job_postings
GROUP BY years_exp
"""
            },
            {
                "schema": """
Table: email_campaigns
Columns: campaign_id (int64), subject (object), content (object), open_rate (float64), click_rate (float64)
Sample rows (5):
  1, 'New Features Release', 'We are excited to announce new AI-powered analytics...', 0.45, 0.12
  2, 'Summer Sale 50% Off', 'Limited time offer on all premium plans...', 0.62, 0.28
  3, 'Product Tutorial Series', 'Learn advanced data visualization techniques...', 0.38, 0.09
  4, 'Customer Success Story', 'How TechCorp improved efficiency with our platform...', 0.51, 0.15
  5, 'Security Update Notice', 'Important security patches and best practices...', 0.71, 0.22
""",
                "question": "What are the top 3 most engaging campaigns about product features or updates?",
                "saber_sql": """
SELECT campaign_id,
       subject,
       SEM_SELECT('Extract main topic: features, tutorial, sale, or security without explanation') AS topic,
       open_rate,
       click_rate
FROM email_campaigns
WHERE SEM_WHERE('This campaign discusses product features, updates, or new capabilities')
ORDER BY click_rate DESC
LIMIT 3
"""
            }
        ],
    }
    
    SHARED_GUIDELINES = """
Guidelines:
- Generate SABER-SQL query using provided table data (columns/rows from schema)
- Use {db_type} SQL syntax for structured operations (numeric comparisons, exact matches, ID joins)
- PRIORITIZE existing structured columns (numeric IDs, categorical codes, status fields) over extracting from text
- Use semantic functions ONLY when semantic/unstructured processing is required or structured data unavailable
- ALWAYS include FROM clause with actual table name(s)
- CRITICAL: Ensure semantic functions reference valid columns from the schema - check spelling and capitalization
- CRITICAL: Never use backticks, quotes, or other punctuation around column names in semantic function prompts
- {db_constraints}

CRITICAL Semantic Function Principles:

1. SEM_SELECT - Extraction Principle:
   Core: Specify WHAT to extract + output FORMAT
   - Describe the input field structure
   - State exactly what to extract with clear intent
   - Extract complete values, not abbreviations or codes
   - Specify output format (plain text, date format, numeric)
   - CRITICAL: Return SCALAR values (string/number), NEVER return JSON/dict/list structures
   - Include parsing rules if needed (remove markup, isolate specific part, convert format)
   - For composite data: specify which component (e.g., "extract value before delimiter")
   - End with clear output instruction (e.g., "return as a single number", "return as plain text")

2. SEM_WHERE - Filtering Principle:
   Core: Write complete EVALUATABLE condition
   - State a full conditional that evaluates to true/false per row
   - Be explicit about what constitutes match/true
   - For substring checks: use "contains", "includes", or "mentions" language
   - For categorical filtering: specify exact matching criteria
   - Avoid ambiguous terms without context (e.g., "best" without ranking metric)

3. COUNT - Validation Principle:
   Core: Verify entity EXISTS before counting
   - Use COUNT(*) for all rows, COUNT(DISTINCT column) for unique values
   - If question asks "how many", determine what to count (rows, distinct values, matches)
   - Add WHERE clause to filter for target entities BEFORE counting
   - For frequency questions: count rows matching specific conditions
   - Prefer simple COUNT over semantic aggregation when possible

4. Ranking/Ordering:
   Core: Use ORDER BY with LIMIT for "top", "first", "highest"
   - For "best ranking": ORDER BY rank_column ASC (lower number = better position)
   - For "maximum value": ORDER BY value_column DESC
   - If single result expected, combine with LIMIT 1

5. String Parsing:
   Core: Prefer semantic extraction over nested SQL functions
   - Use SEM_SELECT with clear extraction prompt and examples
   - Avoid complex SUBSTRING_INDEX nesting
   - For dates/numbers in text: specify format in prompt

Output only the SABER SQL query, no explanations or comments.
"""
    
    SYSTEM_PROMPTS = {        
        'lotus': """
Generate SABER SQL query with LOTUS backend.

Template Syntax: {{column_name}} or {{table.column_name}} for JOINs
Never use quotes, backticks, or punctuation around column names in templates.

Semantic Functions:

1. SEM_WHERE('prompt', 'lotus')
   CRITICAL: MUST include {{column}} reference
   Write evaluatable condition referencing columns via {{column}}
   Example: WHERE SEM_WHERE('{{review_text}} discusses build quality or durability', 'lotus')

2. SEM_SELECT('prompt', 'lotus') AS alias
   CRITICAL: MUST include {{column}} reference for extraction source
   Specify what to extract and output format using {{column}}
   Example: SELECT SEM_SELECT('Extract years of experience from {{description}} as a number without explanation', 'lotus') AS years_exp

3. SEM_AGG('prompt', 'lotus') AS alias
   CRITICAL: MUST include {{column}} reference for aggregation source
   Define aggregation with {{column}} reference
   Example: SELECT SEM_AGG('Summarize {{notes}} in a single sentence', 'lotus') AS summary

4. SEM_ORDER_BY('prompt', 'lotus')
   CRITICAL: MUST include {{column}} reference for ordering source
   Write ordering instruction using {{column}}
   Example: ORDER BY SEM_ORDER_BY('Sort {{details}} by urgency', 'lotus')

5. SEM_GROUP_BY(column, n_groups, 'lotus')
   Example: GROUP BY SEM_GROUP_BY(category, 5, 'lotus')

6. SEM_JOIN(table1, table2, 'prompt', 'lotus')
   CRITICAL: MUST include {{column_of_table1:left}} and {{column_of_table2:right}} references for join keys
   Example: FROM SEM_JOIN(products, desserts, 'the ingredient {{name:left}} is used in {{description:right}}', 'lotus')


Key Rules:
- Use {{column}} for single table, {{table.column}} for JOINs
- When grouping by SEM_SELECT result: GROUP BY the alias, not the function
- Prefer structured columns over semantic extraction when available

{shared_guidelines}
""".strip(),
        
        'docetl': """
Generate SABER SQL query with DocETL backend.

Template Syntax: {{{{ input.column_name }}}} or {{{{ input.table.column_name }}}} for JOINs (Jinja2)
Never use quotes, backticks, or punctuation around column names in templates.

Semantic Functions:

1. SEM_WHERE('prompt', 'docetl')
   CRITICAL: MUST reference column via {{{{ input.column_name }}}}
   Write evaluatable condition. Return true/false explicitly.
   Example: WHERE SEM_WHERE('Review text: {{{{ input.review_text }}}}\n\nReturn true if the review discusses build quality or durability, otherwise false.', 'docetl')

2. SEM_SELECT('prompt', 'docetl') AS alias
   CRITICAL: MUST reference column via {{{{ input.column_name }}}}
   Specify what to extract and output format precisely
   Extract complete values, not codes/abbreviations
   Example: SELECT SEM_SELECT('Description: {{{{ input.description }}}}\n\nExtract the years of experience required (return as a single number) without explanation', 'docetl') AS years_exp

3. SEM_AGG(column, 'prompt', 'docetl') AS alias OR SEM_AGG('prompt', 'docetl') AS alias
   Define aggregation target and operation
   Example: SELECT SEM_AGG(notes, 'Summarize in one sentence', 'docetl') AS summary

4. SEM_ORDER_BY(column, 'prompt', 'docetl')
   Example: ORDER BY SEM_ORDER_BY(details, 'Order details by urgency', 'docetl')

5. SEM_GROUP_BY(column, n_groups, 'docetl')
   Example: GROUP BY SEM_GROUP_BY(category, 5, 'docetl')

6. SEM_JOIN(table1, table2, 'prompt', 'docetl')
   CRITICAL: MUST reference join keys via {{{{ left.table1_column }}}} and {{{{ right.table2_column }}}}
   Example: FROM SEM_JOIN(products, desserts, 'Compare the following product and dessert:\nProduct name: {{{{ left.name }}}}\n\nDessert description: {{{{ right.description }}}}\n\nIs this product used for the dessert? Respond with True if it is a good match or False if it is not a suitable match.', 'docetl')

Key Rules:
- Use {{{{ input.column }}}} for single table, {{{{ input.table.column }}}} for JOINs
- When grouping by SEM_SELECT result: GROUP BY the alias, not the function
- Prefer structured columns over semantic extraction when available

{shared_guidelines}
        """.strip(),

        'palimpzest': """
Generate SABER SQL query with Palimpzest backend.

Template Syntax: Plain natural language (no template markers needed)
Palimpzest has automatic access to all row data.

Semantic Functions:

1. SEM_WHERE('prompt', 'palimpzest')
   Write clear evaluatable condition
   Example: WHERE SEM_WHERE('review text discusses build quality or durability', 'palimpzest')

2. SEM_SELECT('prompt', 'palimpzest') AS alias
   Specify what to extract and format
   Extract complete values, not abbreviations
   Example: SELECT SEM_SELECT('Extract years of experience from the description as a number without explanation', 'palimpzest') AS extracted

3. SEM_ORDER_BY(column, 'prompt', 'palimpzest')
   Write ordering instruction
   Example: ORDER BY SEM_ORDER_BY(details, 'Sort details by urgency', 'palimpzest')

4. SEM_JOIN(table1, table2, 'prompt', 'palimpzest')
   Example: FROM SEM_JOIN(products, desserts, 'the ingredient name from products is used in the description of desserts', 'palimpzest')

Key Rules:
- Use plain natural language (no column markers)
- Prefer structured columns over semantic extraction when available

{shared_guidelines}
        """.strip(),

        'unified': """
Generate SABER SQL query (backend-agnostic).

Semantic Functions:

1. SEM_WHERE('prompt')
   Example: WHERE SEM_WHERE('This row matches the condition')

2. SEM_SELECT('prompt') AS alias
   Example: SELECT SEM_SELECT('Extract value from the field as plain text without explanation') AS extracted

3. SEM_AGG(column, 'prompt') AS alias
4. SEM_ORDER_BY(column, 'prompt')
5. SEM_GROUP_BY(column, n_groups)
6. SEM_JOIN(table1, table2, 'prompt')

{shared_guidelines}
""".strip(),
    }
    
    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None, backend: str = 'lotus', 
                 max_retries: int = 3, fallback_enabled: bool = True, db_type: str = 'duckdb',
                 llm_verification: bool = True):
        """
        Initialize query generator.
        
        Args:
            model: LiteLLM model identifier (e.g., 'openai/gpt-4o-mini', 'hosted_vllm/openai/gpt-oss-20b')
            api_key: API key (optional for local models)
            api_base: API base URL (for local models)
            backend: Target backend for generated queries ('lotus' or 'docetl')
            max_retries: Maximum number of retry attempts for failed queries (default: 3)
            fallback_enabled: Whether to enable fallback to simpler queries (default: True)
            db_type: Type of database ('duckdb', 'mysql', 'sqlite') (default: 'duckdb')
            llm_verification: Enable LLM-powered verification and optimization (default: True)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.backend = backend
        self.max_retries = max_retries
        self.fallback_enabled = fallback_enabled
        self.db_type = db_type
        self.llm_verification = llm_verification
        
        if backend not in ['lotus', 'docetl', 'palimpzest']:
            raise ValueError(f"Backend must be 'lotus', 'docetl', or 'palimpzest', got '{backend}'")
        if db_type not in ['duckdb', 'mysql', 'sqlite']:
            raise ValueError(f"db_type must be 'duckdb', 'mysql', or 'sqlite', got '{db_type}'")
    
    def serialize_dataframe(self, df, table_name: str = "table", max_rows: int = 5) -> str:
        """
        Serialize DataFrame to schema description with sample rows.
        
        Args:
            df: pandas DataFrame
            table_name: Name for the table
            max_rows: Number of sample rows to include (default: 5)
            
        Returns:
            Schema description string with sample data
        """
        schema = f"Table: {table_name}\n"
        schema += f"Columns: {', '.join([f'{col} ({df[col].dtype})' for col in df.columns])}\n"
        schema += f"Sample rows ({min(max_rows, len(df))}):\n"
        for idx, row in df.head(max_rows).iterrows():
            values = ', '.join([repr(v) if isinstance(v, str) else str(v) for v in row.values])
            schema += f"  {values}\n"
        return schema
    
    def _get_db_constraints(self) -> str:
        """Get database-specific constraints and limitations."""
        if self.db_type == 'mysql':
            return (
                "*** CRITICAL MySQL Constraints (STRICTLY ENFORCED) ***\n"
                "- NEVER use POSITION() or SPLIT_PART() - not supported. Use LOCATE() and SUBSTRING_INDEX()\n"
                "- NEVER use FROM dual - not supported\n"
                "- DATEDIFF(date1, date2) requires EXACTLY 2 arguments (not 3)\n"
                "  * For year differences: TIMESTAMPDIFF(YEAR, date1, date2)\n"
                "  * For day differences: DATEDIFF(date2, date1)\n"
                "- ONLY_FULL_GROUP_BY mode: All non-aggregated SELECT columns MUST be in GROUP BY or wrapped in ANY_VALUE()\n"
                "  * Example: SELECT col1, ANY_VALUE(col2) FROM t GROUP BY col1\n"
                "  * With semantic functions: GROUP BY uses result alias, NOT the function\n"
                "  * Example: SELECT SEM_SELECT('extract country', 'lotus') AS country FROM t GROUP BY country\n"
                "- When using COUNT(*) in queries, ensure FROM clause references actual table\n"
                "- Reserved keywords as column names MUST use backticks: `From`, `To`, `When`, `Date`, `Where`, `Group`, `Order`, `Rank`\n"
                "- Avoid nested SUBSTRING_INDEX beyond 1 level - use subqueries/CTEs or SEM_SELECT instead\n"
                "- String comparisons in WHERE: Use exact match or LIKE with wildcards, not approximate\n"
                "- For finding substrings: Use LOCATE(substring, string) > 0 or LIKE '%substring%'\n"
            )
        elif self.db_type == 'duckdb':
            return (
                "*** CRITICAL DuckDB Constraints (STRICTLY ENFORCED) ***\n"
                "- NEVER use SUBSTRING_INDEX, LOCATE, INSTR - these are not DuckDB functions\n"
                "- Use POSITION() for finding, SPLIT_PART() for splitting strings\n"
                "- Use SUBSTRING or SEM_SELECT for text extraction\n"
            )
        elif self.db_type == 'sqlite':
            return (
                "*** CRITICAL SQLite Constraints (STRICTLY ENFORCED) ***\n"
                "- NEVER use SUBSTRING_INDEX, SPLIT_PART - SQLite does NOT support them\n"
                "- Use SUBSTR() and INSTR() for string operations\n"
            )
        return ""
    
    def generate(
        self, 
        question: str,
        tables: List[str] = None,
        schema: str = None,
        df: Dict = None,
        examples: List[Dict] = None,
        backend: str = None,
        max_sample_rows: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 512,
        llm_verification: bool = None
    ) -> str:
        """
        Generate SABER SQL query from natural language.
        
        Args:
            question: Natural language question
            tables: List of table names to include in schema (required if using df dict)
            schema: Database schema description string (alternative to df)
            df: Dict of {table_name: DataFrame} to auto-generate schema
            examples: Optional custom examples (default uses built-in)
            backend: Override default backend ('lotus' or 'docetl')
            max_sample_rows: Number of sample rows to include per table (default: 5)
            temperature: LLM temperature
            max_tokens: Max tokens to generate
            llm_verification: Override LLM verification setting (default uses instance setting)
            
        Returns:
            SABER SQL query string
        """
        # Use specified backend or fall back to default
        backend = backend or self.backend
        if backend not in ['lotus', 'docetl', 'palimpzest', 'unified']:
            raise ValueError(f"Backend must be 'lotus', 'docetl', 'palimpzest', or 'unified', got '{backend}'")
        
        # Generate schema if dataframes provided
        if df and not schema:
            if tables is None:
                raise ValueError("Must specify 'tables' list when using 'df' dictionary")
            
            schema_parts = []
            for table_name in tables:
                if table_name not in df:
                    raise ValueError(f"Table '{table_name}' not found in df dictionary")
                schema_parts.append(self.serialize_dataframe(df[table_name], table_name, max_sample_rows))
            schema = "\n\n".join(schema_parts)
        
        if not schema:
            raise ValueError("Must provide either 'schema' string or 'df' dictionary with 'tables' list")
        
        # Use backend-specific examples if none provided
        if examples is None:
            examples = self.DEFAULT_EXAMPLES[backend]
        
        # Get database-specific constraints
        db_constraints = self._get_db_constraints()
        
        # Format shared guidelines with db_type and constraints
        shared_guidelines = self.SHARED_GUIDELINES.format(db_type=self.db_type, db_constraints=db_constraints)
        
        # Build prompt with backend-specific system message
        if "DeepSeek-R1-Distill-Llama-70B" in self.model:
            messages = [{"role": "user", "content": self.SYSTEM_PROMPTS[backend].format(db_type=self.db_type, db_constraints=db_constraints, shared_guidelines=shared_guidelines)}]
            messages.append({
                "role": "assistant", "content": "Understood. I will generate SABER SQL queries as per the provided instructions."
            })
        else:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPTS[backend].format(db_type=self.db_type, db_constraints=db_constraints, shared_guidelines=shared_guidelines)}]

        # Add examples
        for ex in examples:
            messages.append({
                "role": "user",
                "content": f"Schema:\n{ex['schema']}\n\nQuestion: {ex['question']}"
            })
            messages.append({
                "role": "assistant", 
                "content": ex['saber_sql'].strip()
            })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": f"Schema:\n{schema}\n\nQuestion: {question}"
        })
        
        # Try generation with retries and fallback
        query = self._generate_with_fallback(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            question=question,
            schema=schema,
            backend=backend
        )
        
        # Clean up common formatting issues
        if query.startswith("```sql"):
            query = query[6:]
        if query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
        query = query.strip()
        
        # Normalize column name quotes (backticks to double quotes for DuckDB)
        query = self._normalize_column_quotes(query)
        
        # Apply LLM verification and optimization if enabled
        use_verification = llm_verification if llm_verification is not None else self.llm_verification
        if use_verification:
            query = self._verify_and_optimize_query(
                query=query,
                question=question,
                schema=schema,
                backend=backend,
                temperature=temperature
            )
        
        return query
    
    def _generate_with_fallback(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        question: str,
        schema: str,
        backend: str
    ) -> str:
        """
        Generate query with retry and fallback logic.
        
        Args:
            messages: Message history for LLM
            temperature: Initial temperature
            max_tokens: Max tokens to generate
            question: Original question
            schema: Database schema
            backend: Target backend
            
        Returns:
            Generated SABER SQL query
            
        Raises:
            ValueError: If all retry attempts fail
        """
        last_error = None
        last_query = None
        
        for attempt in range(self.max_retries):
            try:
                retry_temp = temperature + (attempt * 0.1)
                logger.info(f"Query generation attempt {attempt + 1}/{self.max_retries} (temp={retry_temp:.2f})")
                
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=retry_temp,
                    max_tokens=max_tokens,
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                
                if not response or not response.choices or not response.choices[0].message.content:
                    raise ValueError(f"Empty response from LLM: {response}")
                
                query = response.choices[0].message.content.strip()
                logger.debug(f"Raw generated query:\n{query}")
                
                if not query or len(query) < 10:
                    raise ValueError(f"Query too short or empty: '{query}'")
                
                # Auto-fix common issues
                query = self._auto_fix_query(query)
                
                # Apply MySQL-specific fixes
                if self.db_type == 'mysql':
                    query = self._fix_mysql_group_by_issues(query)
                    query = self._fix_scalar_subquery(query)
                
                validation_error = self._verify_query(query, backend)
                if validation_error:
                    logger.warning(f"Generated query that failed validation:\n{query}")
                    raise ValueError(f"Query validation failed: {validation_error}")
                
                if attempt > 0:
                    logger.info(f"Query generation succeeded on attempt {attempt + 1}")
                return query
                
            except Exception as e:
                last_error = e
                last_query = query if 'query' in locals() else None
                logger.warning(f"Query generation attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    break
                
                # Add feedback to prevent same error
                if last_query and 'validation failed' in str(e).lower():
                    messages.append({"role": "assistant", "content": last_query})
                    
                    error_str = str(e).upper()
                    feedback = f"ERROR: {str(e)}\n\n"
                    
                    # ===== Error-Specific Feedback =====
                    
                    if 'MISSING BACKEND PARAMETER' in error_str or 'INSUFFICIENT ARGUMENTS' in error_str:
                        feedback += (
                            f"CRITICAL: Semantic function missing backend parameter.\n\n"
                            f"ALL semantic functions MUST specify the backend parameter: '{backend}'\n\n"
                            f"Correct Format:\n"
                            f"  SEM_WHERE('condition', '{backend}')\n"
                            f"  SEM_SELECT('extraction', '{backend}') AS alias\n"
                            f"  SEM_AGG('aggregation', '{backend}') AS alias\n"
                            f"  SEM_ORDER_BY('sorting criteria', '{backend}')\n\n"
                            f"WRONG: SEM_WHERE('condition')  -- Missing backend\n"
                            f"WRONG: SEM_SELECT('extract')   -- Missing backend and AS alias"
                        )
                    
                    elif 'FROM CLAUSE' in error_str:
                        feedback += (
                            f"CRITICAL: Query missing FROM clause.\n\n"
                            f"FROM clause is REQUIRED when:\n"
                            f"  1. Using WHERE, GROUP BY, ORDER BY, or HAVING\n"
                            f"  2. Using semantic functions (SEM_WHERE, SEM_SELECT, SEM_AGG, etc.)\n"
                            f"  3. Using aggregations (COUNT(*), SUM(column), AVG(column), etc.)\n"
                            f"  4. Accessing any table data or columns\n\n"
                            f"FROM clause is OPTIONAL only for:\n"
                            f"  - Pure literal values: SELECT 46\n"
                            f"  - Arithmetic with literals: SELECT 2025 - 2020\n"
                            f"  - Date functions with literal args: SELECT DATEDIFF('2020-01-01', '2019-01-01')\n\n"
                            f"Fix: Add 'FROM table_name' to your query."
                        )
                    
                    elif 'DATEDIFF' in error_str and self.db_type == 'mysql':
                        feedback += (
                            f"CRITICAL: MySQL DATEDIFF requires EXACTLY 2 date arguments.\n\n"
                            f"Correct Usage:\n"
                            f"  DATEDIFF('2025-12-01', '2025-11-30')          -- Literal dates\n"
                            f"  DATEDIFF(date1_column, date2_column)          -- Column references\n"
                            f"  DATEDIFF(CURDATE(), (SELECT MIN(date) FROM t)) -- Subquery for aggregates\n\n"
                            f"WRONG: DATEDIFF(CURDATE(), MIN(date))  -- Cannot nest aggregate directly\n\n"
                            f"For year calculations: Use TIMESTAMPDIFF(YEAR, date1, date2) instead."
                        )
                    
                    elif 'UNBALANCED' in error_str:
                        feedback += (
                            f"CRITICAL: Unbalanced syntax in query.\n\n"
                            f"Check that ALL opening symbols have matching closing symbols:\n"
                            f"  ( must have )\n"
                            f"  ' must have ' (for strings)\n"
                            f"  {{ must have }} (for LOTUS templates)\n"
                            f"  {{{{ must have }}}} (for DocETL Jinja2 templates)\n\n"
                            f"Common causes:\n"
                            f"  - String with apostrophe: 'Men's walk' (should be 'Men''s walk' or use double quotes)\n"
                            f"  - Unclosed function call: COUNT(\n"
                            f"  - Missing template closing: {{column\n\n"
                            f"Fix: Balance all symbols and properly escape strings."
                        )
                    
                    elif 'NUMBER TABLE' in error_str or 'NUMBERS.N' in error_str or 'AUXILIARY' in error_str:
                        feedback += (
                            f"CRITICAL: Query uses auxiliary number table pattern (not allowed).\n\n"
                            f"WRONG Patterns:\n"
                            f"  JOIN (SELECT 1 AS n UNION SELECT 2 UNION ...) numbers ON ...\n"
                            f"  FROM numbers WHERE numbers.n <= 10\n\n"
                            f"Instead, use:\n"
                            f"  - Direct operations on existing table data\n"
                            f"  - Semantic functions for data extraction/transformation\n"
                            f"  - Standard SQL operations without generating number sequences"
                        )
                    
                    elif 'RESERVED KEYWORD' in error_str or 'BACKTICKS' in error_str or ('FROM' in error_str and 'COLUMN' in error_str):
                        feedback += (
                            f"CRITICAL: Column name is a {self.db_type.upper()} reserved keyword.\n\n"
                            f"Reserved keywords as column names MUST be quoted:\n"
                            f"  MySQL: Use backticks `From`, `To`, `When`, `Date`\n"
                            f"  DuckDB/SQLite: Use double quotes \"From\", \"To\", \"When\", \"Date\"\n\n"
                            f"Examples:\n"
                            f"  WRONG: SELECT From, To FROM table WHERE From = '2020'\n"
                            f"  CORRECT (MySQL): SELECT `From`, `To` FROM table WHERE `From` = '2020'\n"
                            f"  CORRECT (DuckDB): SELECT \"From\", \"To\" FROM table WHERE \"From\" = '2020'\n\n"
                            f"Note: Auto-fix should have handled this, but check your query carefully."
                        )
                    
                    elif 'SUBSTRING_INDEX' in error_str or 'SPLIT_PART' in error_str or 'INSTR' in error_str or 'POSITION' in error_str:
                        feedback += (
                            f"CRITICAL: Query uses incompatible string function for {self.db_type}.\n\n"
                            f"Database-Specific Functions:\n"
                            f"  DuckDB: SPLIT_PART(), POSITION(), SUBSTRING()\n"
                            f"  MySQL: SUBSTRING_INDEX(), LOCATE(), SUBSTRING()\n"
                            f"  SQLite: SUBSTR(), INSTR()\n\n"
                            f"RECOMMENDED: Use SEM_SELECT for text extraction instead.\n"
                            f"  Example: SEM_SELECT('Extract first word from {{text}}', '{backend}') AS first_word\n\n"
                            f"This avoids database-specific syntax and is more semantic."
                        )
                    
                    elif 'REGEXP_EXTRACT' in error_str:
                        feedback += (
                            f"CRITICAL: REGEXP_EXTRACT not supported in {self.db_type}.\n\n"
                            f"Use SEM_SELECT for text extraction:\n"
                            f"  SEM_SELECT('Extract pattern from {{text}}', '{backend}') AS extracted\n\n"
                            f"This is more flexible and works across all backends."
                        )
                    
                    elif 'GROUP BY' in error_str and ('NOT IN' in error_str or 'ONLY_FULL_GROUP_BY' in error_str):
                        feedback += (
                            f"CRITICAL: MySQL ONLY_FULL_GROUP_BY mode violation.\n\n"
                            f"ALL non-aggregated SELECT columns MUST be in GROUP BY or wrapped in aggregates.\n\n"
                            f"Wrong:\n"
                            f"  SELECT col1, col2, COUNT(*) FROM t GROUP BY col1\n"
                            f"  (col2 is not aggregated and not in GROUP BY)\n\n"
                            f"Correct Options:\n"
                            f"  1. Add to GROUP BY: SELECT col1, col2, COUNT(*) FROM t GROUP BY col1, col2\n"
                            f"  2. Use ANY_VALUE(): SELECT col1, ANY_VALUE(col2), COUNT(*) FROM t GROUP BY col1\n"
                            f"  3. Remove from SELECT: SELECT col1, COUNT(*) FROM t GROUP BY col1\n\n"
                            f"With semantic functions:\n"
                            f"  SELECT SEM_SELECT('extract', '{backend}') AS result, COUNT(*)\n"
                            f"  FROM t\n"
                            f"  GROUP BY result  -- Use the alias, not the full function"
                        )
                    
                    elif 'COLUMN REFERENCE ERROR' in error_str or 'INVALID COLUMN' in error_str:
                        feedback += (
                            f"CRITICAL: Semantic function references non-existent column.\n\n"
                            f"Check the schema carefully. Column names:\n"
                            f"  - Are case-sensitive in some databases\n"
                            f"  - Must match exactly (no typos)\n"
                            f"  - Should not include backticks, quotes, or other punctuation in templates\n\n"
                            f"Template Syntax by Backend:\n"
                            f"  LOTUS: {{column_name}} or {{table.column_name}}\n"
                            f"  DocETL: {{{{ input.column_name }}}} or {{{{ input.table.column_name }}}}\n"
                            f"  Palimpzest: No column references needed (auto-access)\n\n"
                            f"WRONG: {{`column`}}, {{{{ input.`column` }}}}\n"
                            f"CORRECT: {{column}}, {{{{ input.column }}}}"
                        )
                    
                    elif 'SYNTAX ERROR' in error_str and 'SUBSTRING_INDEX' in error_str:
                        feedback += (
                            f"CRITICAL: Complex nested SUBSTRING_INDEX causing syntax error.\n\n"
                            f"Problem: Nested SUBSTRING_INDEX in arithmetic expressions is error-prone.\n\n"
                            f"WRONG:\n"
                            f"  SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(col, ' ', 1), '-', 2) * 10 FROM t\n\n"
                            f"Better Approaches:\n"
                            f"  1. Use subquery/CTE:\n"
                            f"     WITH extracted AS (\n"
                            f"       SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(col, ' ', 1), '-', 2) AS val FROM t\n"
                            f"     )\n"
                            f"     SELECT val * 10 FROM extracted\n\n"
                            f"  2. Use SEM_SELECT (RECOMMENDED):\n"
                            f"     SELECT SEM_SELECT('Extract numeric value from {{col}}', '{backend}') AS val FROM t"
                        )
                    
                    elif 'BACKTICKS' in error_str and backend in ['lotus', 'docetl']:
                        feedback += (
                            f"CRITICAL: Backticks not allowed in {backend.upper()} semantic function templates.\n\n"
                            f"Column references in semantic prompts must use:\n"
                            f"  LOTUS: {{column}} NOT {{`column`}}\n"
                            f"  DocETL: {{{{ input.column }}}} NOT {{{{ input.`column` }}}}\n\n"
                            f"Backticks are for SQL column quoting, NOT for template column references.\n\n"
                            f"Fix: Remove all backticks/quotes from inside template curly braces."
                        )
                    
                    elif 'PROMPT' in error_str and 'VAGUE' in error_str:
                        feedback += (
                            f"WARNING: Semantic function prompt is too vague.\n\n"
                            f"For {backend.upper()} backend, prompts should be:\n"
                            f"  - Clear and specific\n"
                            f"  - Include action words (Extract, Return, Check, Identify, etc.)\n"
                            f"  - Specify output format when needed (as a number, as YYYY-MM-DD, as true/false)\n\n"
                            f"Examples:\n"
                            f"  VAGUE: SEM_SELECT('name', '{backend}')\n"
                            f"  CLEAR: SEM_SELECT('Extract the person name as plain text', '{backend}')\n\n"
                            f"  VAGUE: SEM_WHERE('relevant', '{backend}')\n"
                            f"  CLEAR: SEM_WHERE('This text is relevant to machine learning', '{backend}')"
                        )
                    
                    else:
                        # Generic feedback
                        feedback += (
                            f"Generate a valid SABER SQL query for {self.db_type}.\n\n"
                            f"Requirements:\n"
                            f"  - Use {self.db_type}-compatible SQL syntax\n"
                            f"  - Include '{backend}' backend parameter in all semantic functions\n"
                            f"  - Follow {backend.upper()} template syntax exactly\n"
                            f"  - Ensure FROM clause when accessing table data\n"
                            f"  - Balance all parentheses, quotes, and brackets\n"
                            f"  - Reference only columns that exist in the schema"
                        )
                    
                    messages.append({"role": "user", "content": feedback})
                
                import time
                time.sleep(0.5 * (2 ** attempt))
        
        # All retries failed - try fallback if enabled
        if self.fallback_enabled:
            logger.warning("All retry attempts failed, attempting fallback to simple query")
            try:
                return self._generate_fallback_query(question, schema, backend)
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                # Continue to raise original error
        
        # Provide detailed error message
        error_msg = f"LLM query generation failed after {self.max_retries} attempts. "
        error_msg += f"Last error: {last_error}. "
        error_msg += f"Question: '{question[:100]}...'. "
        error_msg += "Try: 1) Check if LLM endpoint is working, 2) Simplify the question, 3) Increase max_retries."
        
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def _generate_fallback_query(self, question: str, schema: str, backend: str) -> str:
        """
        Generate a simple fallback query when LLM generation fails.
        
        Args:
            question: Original question
            schema: Database schema
            backend: Target backend
            
        Returns:
            Simple SABER SQL query
        """
        # Extract table names from schema
        import re
        table_matches = re.findall(r'Table:\s+(\w+)', schema)
        
        if not table_matches:
            raise ValueError("Cannot generate fallback query - no tables found in schema")
        
        # Use first table as base
        table_name = table_matches[0]
        
        # Extract column names
        columns_match = re.search(rf'Table:\s+{table_name}\s+Columns:\s+([^\n]+)', schema)
        if not columns_match:
            raise ValueError(f"Cannot parse columns for table {table_name}")
        
        # Parse column names (format: "col1 (type), col2 (type), ...")
        columns_str = columns_match.group(1)
        columns = [col.split('(')[0].strip() for col in columns_str.split(',')]
        
        # Find best text column for semantic operations
        text_keywords = ['description', 'text', 'content', 'comment', 'bio', 'abstract', 'note', 'review']
        first_text_col = None
        
        # First try to find a column with a semantic name
        for col in columns:
            if any(keyword in col.lower() for keyword in text_keywords):
                first_text_col = col
                break
        
        # Otherwise, find any TEXT/VARCHAR column
        if not first_text_col:
            for col in columns:
                col_def = [part for part in columns_str.split(',') if col in part]
                if col_def and any(t in col_def[0].upper() for t in ['TEXT', 'VARCHAR', 'STRING']):
                    first_text_col = col
                    break
        
        # Fallback to first column
        if not first_text_col:
            first_text_col = columns[0]
        
        # Generate simple query based on backend
        if backend == 'lotus':
            # Simple semantic WHERE query
            query = f"""
SELECT *
FROM {table_name}
WHERE SEM_WHERE('{{{first_text_col}}} is relevant to the question', 'lotus')
LIMIT 10
""".strip()
        
        elif backend == 'docetl':
            # Simple docetl query
            query = f"""
SELECT *
FROM {table_name}
WHERE SEM_WHERE('Content: {{{{ input.{first_text_col} }}}}\n\nReturn true if relevant to the question.', 'docetl')
LIMIT 10
""".strip()
        
        else:  # palimpzest
            # Simple palimpzest query
            query = f"""
SELECT *
FROM {table_name}
WHERE SEM_WHERE('This row is relevant to the question', 'palimpzest')
LIMIT 10
""".strip()
        
        logger.info(f"Generated fallback query: {query}")
        return query
    
    def _auto_fix_query(self, query: str) -> str:
        """
        Auto-fix common query issues before verification.
        
        Fixes applied:
        1. MySQL reserved keywords - Add backticks
        2. Semantic function column references with punctuation - Remove backticks/quotes
        3. Simple text comparison patterns - Suggest semantic functions
        
        Args:
            query: SQL query to fix
            
        Returns:
            Fixed SQL query
        """
        import re
        
        # ===== FIX 1: MySQL Reserved Keywords =====
        if self.db_type == 'mysql':
            reserved_keywords = [
                'From', 'To', 'When', 'Date', 'Where', 'Group', 'Order', 'By', 
                'Select', 'Values', 'Limit', 'Rank', 'Index', 'Key', 'Action', 
                'Status', 'Duration', 'Type', 'Time', 'Year', 'Month', 'Day'
            ]
            
            for keyword in reserved_keywords:
                # Skip if already quoted
                if f'`{keyword}`' in query or f'"{keyword}"' in query:
                    continue
                
                # Patterns for reserved keyword usage without quotes
                patterns = [
                    # SELECT keyword (not followed by opening paren for functions)
                    (rf'\bSELECT\s+({keyword})\b(?!\s*\()', rf'SELECT `{keyword}`'),
                    # , keyword (in SELECT list, not followed by paren or AS)
                    (rf',\s*({keyword})\b(?!\s*\(|\s*AS)', rf', `{keyword}`'),
                    # keyword = value (WHERE conditions)
                    (rf'\b({keyword})\s*([=<>!]+)', rf'`{keyword}` \2'),
                    # ORDER BY keyword
                    (rf'ORDER\s+BY\s+({keyword})\b', rf'ORDER BY `{keyword}`'),
                    # GROUP BY keyword
                    (rf'GROUP\s+BY\s+({keyword})\b', rf'GROUP BY `{keyword}`'),
                    # keyword IN (...)
                    (rf'\b({keyword})\s+IN\s*\(', rf'`{keyword}` IN ('),
                    # keyword IS NULL/NOT NULL
                    (rf'\b({keyword})\s+IS\s+(NOT\s+)?NULL', rf'`{keyword}` IS \2NULL'),
                    # WHERE/AND/OR keyword =
                    (rf'\b(WHERE|AND|OR|ON|WHEN|THEN|ELSE|HAVING)\s+({keyword})\b(?=\s*[=<>!])', rf'\1 `{keyword}`'),
                    # DISTINCT keyword
                    (rf'DISTINCT\s+({keyword})\b(?!\s*\()', rf'DISTINCT `{keyword}`'),
                    # AS keyword (aliasing as reserved word)
                    (rf'AS\s+({keyword})\b', rf'AS `{keyword}`'),
                    # COUNT/SUM/AVG/MAX/MIN(keyword)
                    (rf'(COUNT|SUM|AVG|MAX|MIN|ANY_VALUE)\s*\(\s*({keyword})\s*\)', rf'\1(`{keyword}`)'),
                ]
                
                for pat, replacement in patterns:
                    if re.search(pat, query, re.IGNORECASE):
                        query = re.sub(pat, replacement, query, flags=re.IGNORECASE)
                        logger.info(f"Auto-fixed reserved keyword '{keyword}' with backticks")
        
        # ===== FIX 2: Remove Punctuation from Semantic Function Column References =====
        # LOTUS: {`column`} -> {column}
        if self.backend == 'lotus' or 'lotus' in query.lower():
            # Pattern: {`column`} or {table.`column`}
            backtick_in_template = re.findall(r'\{([^}]*`[^}]*)\}', query)
            for match in backtick_in_template:
                fixed = match.replace('`', '').replace('"', '')
                query = query.replace('{' + match + '}', '{' + fixed + '}')
                logger.info(f"Removed backticks from LOTUS template: {{{match}}} -> {{{fixed}}}")
        
        # DocETL: {{ input.`column` }} -> {{ input.column }}
        if self.backend == 'docetl' or 'docetl' in query.lower():
            # Pattern: {{ input.`column` }} or {{ input.table.`column` }}
            backtick_in_jinja = re.findall(r'\{\{\s*input\.([^}]*`[^}]*)\s*\}\}', query)
            for match in backtick_in_jinja:
                fixed = match.replace('`', '').replace('"', '')
                query = re.sub(
                    r'\{\{\s*input\.' + re.escape(match) + r'\s*\}\}',
                    '{{ input.' + fixed + ' }}',
                    query
                )
                logger.info(f"Removed backticks from DocETL template: {{{{ input.{match} }}}} -> {{{{ input.{fixed} }}}}")
        
        # ===== FIX 3: Complex Nested String Functions -> Suggest SEM_SELECT =====
        # Detect complex nested SUBSTRING_INDEX patterns that might cause errors
        complex_substring_patterns = [
            r'SUBSTRING_INDEX\s*\(\s*SUBSTRING_INDEX\s*\([^)]+\)\s*,[^,]+,[^)]+\)\s*[-+*/]',  # Nested in arithmetic
            r'SUBSTRING_INDEX\s*\(\s*SUBSTRING_INDEX\s*\(\s*SUBSTRING_INDEX',  # Triple nesting
        ]
        
        for pattern in complex_substring_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(
                    "Query contains complex nested string functions that may cause errors. "
                    "Consider using SEM_SELECT for text extraction instead of complex SQL string parsing."
                )
                break
        
        return query
    
    def _verify_query(self, query: str, backend: str) -> Optional[str]:
        """
        Verify that generated query has valid structure and syntax.
        
        Validation stages:
        1. Basic SQL structure (SELECT, FROM)
        2. Syntax balance (parentheses, quotes)
        3. Database-specific constraints (DATEDIFF, functions)
        4. Semantic function validation (backend, arguments, templates)
        5. Error indicators and malformed content
        
        Args:
            query: Generated SQL query
            backend: Target backend
            
        Returns:
            Error message if validation fails, None if valid
        """
        import re
        
        query_upper = query.upper()
        
        # ===== STAGE 1: Basic SQL Structure =====
        
        # Must contain SELECT
        if 'SELECT' not in query_upper:
            return "Query must contain SELECT statement"
        
        # Must contain FROM clause unless it's a literal expression (no table data needed)
        if 'FROM' not in query_upper:
            # Check if query requires table data access
            # Patterns that REQUIRE FROM clause:
            requires_from_patterns = [
                (r'\bCOUNT\s*\(\s*\*\s*\)', 'COUNT(*) requires table rows'),
                (r'\b(SUM|AVG|MIN|MAX)\s*\(\s*\w+\s*\)', 'Aggregate functions on columns require FROM'),
                (r'\bCOUNT\s*\(\s*DISTINCT\s+\w+\s*\)', 'COUNT(DISTINCT column) requires FROM'),
                (r'\bSEM_(WHERE|SELECT|AGG|ORDER_BY|GROUP_BY|JOIN)\s*\(', 'Semantic functions require FROM'),
                (r'\bWHERE\b', 'WHERE clause requires FROM'),
                (r'\bGROUP\s+BY\b', 'GROUP BY requires FROM'),
                (r'\bORDER\s+BY\b(?!\s+SEM_ORDER_BY)', 'ORDER BY requires FROM'),
                (r'\bJOIN\b', 'JOIN requires FROM'),
                (r'\bHAVING\b', 'HAVING requires FROM'),
            ]
            
            for pattern, reason in requires_from_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return f"Query must contain FROM clause: {reason}"
            
            # Allow pure literal expressions (constants, calculations with literals, date functions with literal args)
            # Valid examples:
            #   SELECT 46
            #   SELECT 'George W. Bush'
            #   SELECT DATEDIFF('2008-11-01', '2008-10-31')
            #   SELECT 1998 - 1952
            #   SELECT CURRENT_DATE()
            #   SELECT 2020 AS year, 'Result' AS label
            
            # Extract SELECT clause content
            select_match = re.search(r'SELECT\s+(.+?)(?:;|\s*$)', query, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return "Invalid SELECT statement structure"
            
            select_content = select_match.group(1).strip()
            
            # Check if SELECT contains only literal expressions
            # Split by commas (not inside parentheses or quotes)
            depth = 0
            in_string = False
            string_char = None
            columns = []
            current_col = []
            
            for char in select_content:
                if char in ('"', "'") and not in_string:
                    in_string = True
                    string_char = char
                elif in_string and char == string_char:
                    in_string = False
                elif not in_string:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                    elif char == ',' and depth == 0:
                        columns.append(''.join(current_col).strip())
                        current_col = []
                        continue
                current_col.append(char)
            
            if current_col:
                columns.append(''.join(current_col).strip())
            
            # Check each column expression
            for col_expr in columns:
                # Remove AS alias if present
                col_without_alias = re.sub(r'\s+AS\s+\w+\s*$', '', col_expr, flags=re.IGNORECASE).strip()
                
                # Allowed literal patterns:
                # 1. Numbers: 23, 3.14, -10
                # 2. Strings: 'text', "text"
                # 3. Arithmetic with literals: 2025 - 1925, (10 + 5) * 2
                # 4. Date functions with literal args: DATEDIFF('2025-01-01', '2020-01-01')
                # 5. System functions: CURRENT_DATE(), NOW()
                
                is_literal = (
                    # Pure number
                    re.match(r'^-?[\d\.]+$', col_without_alias) or
                    # String literal
                    re.match(r"^'[^']*'$", col_without_alias) or
                    re.match(r'^"[^"]*"$', col_without_alias) or
                    # Arithmetic expression with numbers/literals
                    re.match(r'^[\d\.\s+\-*/()]+$', col_without_alias) or
                    # Date functions with literal arguments
                    re.match(r'^(DATEDIFF|TIMESTAMPDIFF|DATE_ADD|DATE_SUB)\s*\([\'"\d\s\-,()]+\)$', col_without_alias, re.IGNORECASE) or
                    # System functions (no args or no column refs)
                    re.match(r'^(CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP|NOW|CURDATE|CURTIME)\s*\(\s*\)$', col_without_alias, re.IGNORECASE)
                )
                
                if not is_literal:
                    return f"Query must contain FROM clause when accessing table data or using non-literal expressions"
            
            # All columns are literals - allow without FROM
            logger.debug(f"Allowed SELECT without FROM: pure literal expression")
            return None
        
        # ===== STAGE 2: Syntax Balance =====
        
        # Check for balanced parentheses and quotes (excluding those inside strings)
        paren_balance = 0
        in_single_quote = False
        in_double_quote = False
        i = 0
        
        while i < len(query):
            ch = query[i]
            
            # Handle escape sequences
            if ch == '\\' and i + 1 < len(query):
                i += 2
                continue
            
            # Track string context
            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            # Count parentheses only outside strings
            elif not in_single_quote and not in_double_quote:
                if ch == '(':
                    paren_balance += 1
                elif ch == ')':
                    paren_balance -= 1
            i += 1
        
        if paren_balance != 0:
            return f"Unbalanced parentheses (balance: {paren_balance})"
        
        if in_single_quote or in_double_quote:
            return "Unbalanced quotes"
        
        # ===== STAGE 3: Basic Structure Validation =====
        
        # Check for Oracle-style FROM dual in MySQL
        if self.db_type == 'mysql' and re.search(r'FROM\s+dual', query, re.IGNORECASE):
            return "MySQL does not support 'FROM dual'. Queries without tables should use a real table or be restructured."
        
        # ===== STAGE 4: Database-Specific Constraints =====
        
        # Warn about potentially unsupported functions (non-blocking)
        # Note: DatabaseAdapter.normalize_sql_for_db() handles normalization
        if self.db_type == 'duckdb':
            potentially_unsupported = ['SUBSTRING_INDEX', 'REGEXP_EXTRACT', 'STR_TO_DATE', 'DATE_FORMAT', 'INSTR']
        elif self.db_type == 'mysql':
            potentially_unsupported = ['SPLIT_PART', 'POSITION(']
        else:  # sqlite
            potentially_unsupported = ['SUBSTRING_INDEX', 'SPLIT_PART', 'REGEXP_EXTRACT']
        
        for func in potentially_unsupported:
            if func in query_upper:
                logger.warning(f"Query contains potentially unsupported function '{func}' for {self.db_type}. "
                              f"DatabaseAdapter will attempt normalization, but consider using semantic functions if execution fails.")
        
        # MySQL DATEDIFF requires exactly 2 arguments
        if self.db_type == 'mysql' and 'DATEDIFF' in query_upper:
            # Find all DATEDIFF calls with proper multi-level nesting handling
            i = 0
            while i < len(query):
                # Look for DATEDIFF(
                if re.match(r'DATEDIFF\s*\(', query[i:], re.IGNORECASE):
                    match_obj = re.match(r'DATEDIFF\s*\(', query[i:], re.IGNORECASE)
                    start_pos = i + match_obj.end() - 1  # Position of opening (
                    
                    # Extract content within balanced parentheses
                    depth = 1
                    j = start_pos + 1
                    while j < len(query) and depth > 0:
                        if query[j] == '(':
                            depth += 1
                        elif query[j] == ')':
                            depth -= 1
                        j += 1
                    
                    if depth == 0:
                        func_content = query[start_pos + 1:j - 1]
                        
                        # Count commas at depth 0 (not inside nested parentheses or strings)
                        depth = 0
                        comma_count = 0
                        in_string = False
                        string_char = None
                        
                        for char in func_content:
                            if char in ('"', "'") and not in_string:
                                in_string = True
                                string_char = char
                            elif in_string and char == string_char:
                                in_string = False
                            elif not in_string:
                                if char == '(':
                                    depth += 1
                                elif char == ')':
                                    depth -= 1
                                elif char == ',' and depth == 0:
                                    comma_count += 1
                        
                        if comma_count != 1:
                            # Extract a preview of the problematic call
                            preview = query[i:min(i + 70, len(query))]
                            return f"MySQL DATEDIFF requires exactly 2 arguments: DATEDIFF(date1, date2). Found {comma_count + 1} arguments in '{preview}...'."
                    
                    i = j
                else:
                    i += 1
        
        # Check for number table patterns (auxiliary sequences)
        number_table_patterns = [
            r'JOIN\s*\(\s*SELECT.*?\bAS\s+n\b',  # JOIN (SELECT ... AS n
            r'numbers\.n',  # numbers.n reference
            r'SELECT\s+\d+\s+AS\s+n\s+UNION',  # SELECT 1 AS n UNION SELECT 2
        ]
        for pattern in number_table_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return "Query uses number table pattern (auxiliary sequences). Use direct operations or semantic functions instead."
        
        # MySQL reserved keyword validation (auto-fix moved to _auto_fix_query)
        if self.db_type == 'mysql':
            # Check for unquoted reserved keywords used as column names
            reserved_keywords = ['From', 'To', 'When', 'Date', 'Where', 'Group', 'Order', 'By', 'Select', 'Values', 'Limit', 'Rank', 'Index', 'Key', 'Action', 'Status']
            for keyword in reserved_keywords:
                # Skip if already quoted
                if f'`{keyword}`' in query or f'"{keyword}"' in query:
                    continue
                
                # Just warn if we find them, as auto-fix should have handled it
                # Pattern: keyword used as column
                if re.search(rf'\b{keyword}\b', query, re.IGNORECASE):
                     # This is a loose check, might false positive on SQL keywords, but it's just a warning/check
                     pass
        
        # ===== STAGE 5: Semantic Function Validation =====
        
        semantic_funcs = ['SEM_WHERE', 'SEM_SELECT', 'SEM_AGG', 'SEM_ORDER_BY', 'SEM_GROUP_BY', 'SEM_JOIN']
        
        for func in semantic_funcs:
            if func in query_upper:
                pattern = rf'{func}\s*\([^)]*\)'
                matches = re.finditer(pattern, query, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    func_call = match.group(0)
                    
                    # Check backend parameter (unified mode doesn't require it)
                    if backend != 'unified' and f"'{backend}'" not in func_call and f'"{backend}"' not in func_call:
                        return f"{func} call missing or incorrect backend parameter: expected '{backend}'"
                    
                    # Count arguments (commas outside of quotes)
                    in_quotes = False
                    paren_depth = 0
                    comma_count = 0
                    
                    for char in func_call:
                        if char == "'" or char == '"':
                            in_quotes = not in_quotes
                        elif char == '(' and not in_quotes:
                            paren_depth += 1
                        elif char == ')' and not in_quotes:
                            paren_depth -= 1
                        elif char == ',' and not in_quotes and paren_depth == 1:
                            comma_count += 1
                    
                    # Determine minimum required arguments
                    min_commas = 1  # Default: arg + backend
                    if func == 'SEM_GROUP_BY':
                        min_commas = 2  # column, n_groups, backend
                    elif func in ['SEM_ORDER_BY', 'SEM_AGG']:
                        min_commas = 1  # Can have 2 or 3 args
                    
                    if backend == 'unified':
                        min_commas -= 1  # No backend arg
                    
                    if comma_count < min_commas:
                        # Check for common error: missing second argument or backend parameter
                        if func == 'SEM_WHERE' and comma_count == 0:
                            return f"{func} missing backend parameter. Format: SEM_WHERE('condition', '{backend}')"
                        return f"{func} has insufficient arguments (expected at least {min_commas + 1})"
        
        # MySQL GROUP BY with semantic functions validation
        if self.db_type == 'mysql' and 'GROUP BY' in query_upper:
            # Check if GROUP BY uses semantic function but SELECT doesn't use same expression
            group_by_match = re.search(r'GROUP\s+BY\s+(SEM_\w+\([^)]+\))', query, re.IGNORECASE)
            if group_by_match:
                group_expr = group_by_match.group(1)
                # Verify SELECT clause uses aggregates or the same semantic function
                select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1)
                    # Check if non-aggregate columns exist
                    if group_expr not in select_clause:
                        # This might cause GROUP BY errors - log warning but don't block
                        logger.warning(f"GROUP BY uses {group_expr} but SELECT clause may not match exactly")
            
            # Check for non-aggregated columns in SELECT that aren't in GROUP BY
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # Extract all selected columns that aren't aggregate functions
                column_pattern = r'(?:^|,)\s*(?!.*(?:COUNT|SUM|AVG|MAX|MIN|SEM_AGG)\s*\()(\w+(?:\.\w+)?)\s*(?:AS\s+\w+)?'
                selected_cols = re.findall(column_pattern, select_clause, re.IGNORECASE)
                
                group_by_match = re.search(r'GROUP\s+BY\s+(.+?)(?:ORDER|HAVING|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
                if group_by_match:
                    group_by_clause = group_by_match.group(1).strip()
                    
                    # Check if all non-aggregated columns are in GROUP BY
                    for col in selected_cols:
                        col = col.strip()
                        if col and col.lower() not in ['*', 'distinct'] and col not in group_by_clause:
                            logger.warning(f"Column '{col}' in SELECT but not in GROUP BY - may cause MySQL strict mode error")
        
        # Check for potential empty column issues in aggregate contexts
        if 'COUNT(DISTINCT SEM_' in query_upper or 'COUNT(SEM_' in query_upper:
            # Pattern: COUNT(DISTINCT SEM_SELECT(...)) without AS alias
            pattern = r'COUNT\s*\(\s*(?:DISTINCT\s+)?SEM_\w+\s*\([^)]+\)\s*\)(?!\s+AS\s+\w+)'
            if re.search(pattern, query, re.IGNORECASE):
                return "COUNT with SEM_SELECT must provide an explicit alias. Use: SELECT COUNT(...) AS alias_name"
        
        # Check for nested semantic functions without proper aliasing
        if 'SEM_SELECT' in query_upper:
            # Check if SEM_SELECT is used in aggregations or subqueries without AS alias
            pattern = r'SEM_SELECT\s*\([^)]+\)(?!\s+AS\s+\w+)'
            unaliased_matches = list(re.finditer(pattern, query, re.IGNORECASE))
            
            for match in unaliased_matches:
                # Check if this SEM_SELECT is inside an aggregate or function call
                context_start = max(0, match.start() - 50)
                context = query[context_start:match.end()]
                if re.search(r'(COUNT|MAX|MIN|SUM|AVG|DISTINCT)\s*\(.*SEM_SELECT', context, re.IGNORECASE):
                    return "SEM_SELECT inside aggregate functions must have its own AS alias before aggregation"
        
        # ===== STAGE 6: Backend-Specific Template Syntax Validation =====
        
        # LOTUS Backend Validation
        if backend == 'lotus':
            if any(func in query_upper for func in ['SEM_WHERE', 'SEM_SELECT', 'SEM_AGG', 'SEM_ORDER_BY', 'SEM_JOIN']):
                # Extract semantic function prompts
                pattern = r"SEM_\w+\s*\(['\"]([^'\"]*)['\"]"
                matches = re.findall(pattern, query, re.IGNORECASE)
                
                # Check for template syntax
                has_template = any('{' in m and '}' in m for m in matches if m)
                if matches and not has_template:
                    logger.warning(
                        "LOTUS query missing {column} template syntax. "
                        "Semantic functions should reference columns using {column_name} syntax."
                    )
                
                # CRITICAL: Check for backticks/quotes in column references
                invalid_patterns = [
                    (r'\{\s*`[^}]+`\s*\}', 'backticks'),
                    (r'\{\s*"[^}]+"s*\}', 'double quotes'),
                    (r"\{\s*'[^}]+'\s*\}", 'single quotes'),
                ]
                for pattern, quote_type in invalid_patterns:
                    if re.search(pattern, query):
                        return f"LOTUS templates must NOT use {quote_type} around column names. Use {{column}} not {{{quote_type}column{quote_type}}}"
                
                # # Check for proper table alias usage with JOINs
                # if 'JOIN' in query_upper or re.search(r'FROM\s+\w+\s+(?:AS\s+)?\w+\s*,', query, re.IGNORECASE):
                #     # Has table aliases or multiple tables
                #     # Check if templates use table.column syntax
                #     template_refs = re.findall(r'\{([^}]+)\}', query)
                #     has_qualified = any('.' in ref for ref in template_refs)
                #     if template_refs and not has_qualified:
                #         logger.warning(
                #             "LOTUS query with JOINs/aliases should use {table.column} syntax for clarity"
                #         )
        
        # DocETL Backend Validation
        elif backend == 'docetl':
            if any(func in query_upper for func in ['SEM_WHERE', 'SEM_SELECT', 'SEM_JOIN']):
                # Extract semantic function prompts
                pattern = r"SEM_\w+\s*\(['\"]([^'\"]*)['\"]"
                matches = re.findall(pattern, query, re.IGNORECASE | re.DOTALL)
                
                # Check for Jinja2 syntax
                has_jinja = any('{{' in m and '}}' in m for m in matches if m)
                if matches and not has_jinja:
                    logger.warning(
                        "DocETL query missing {{ input.column }} Jinja2 syntax. "
                        "Semantic functions should reference columns using {{ input.column_name }} syntax."
                    )
                
                # CRITICAL: Check for backticks/quotes in column references
                invalid_patterns = [
                    (r'\{\{\s*input\.`[^}]+`\s*\}\}', 'backticks'),
                    (r'\{\{\s*input\."[^}]+"\s*\}\}', 'double quotes'),
                    (r"\{\{\s*input\.'[^}]+'\s*\}\}", 'single quotes'),
                ]
                for pattern, quote_type in invalid_patterns:
                    if re.search(pattern, query):
                        return f"DocETL templates must NOT use {quote_type} around column names. Use {{{{ input.column }}}} not {{{{ input.{quote_type}column{quote_type} }}}}"
                
                # Check for proper table alias usage with JOINs
                # if 'JOIN' in query_upper or re.search(r'FROM\s+\w+\s+(?:AS\s+)?\w+\s*,', query, re.IGNORECASE):
                #     # Has table aliases - should use {{ input.table.column }}
                #     jinja_refs = re.findall(r'\{\{\s*input\.([^}]+)\s*\}\}', query)
                #     has_qualified = any('.' in ref for ref in jinja_refs)
                #     if jinja_refs and not has_qualified:
                #         logger.warning(
                #             "DocETL query with JOINs/aliases should use {{ input.table.column }} syntax"
                #         )
        
        # Palimpzest Backend Validation
        elif backend == 'palimpzest':
            # SEM_ORDER_BY must have column as first argument
            if 'SEM_ORDER_BY' in query_upper:
                pattern = r'SEM_ORDER_BY\s*\(\s*([^,\)]+)\s*,'
                matches = re.findall(pattern, query, re.IGNORECASE)
                if not matches:
                    return "Palimpzest SEM_ORDER_BY must specify column as first argument: SEM_ORDER_BY(column, 'prompt', 'palimpzest')"
                
                first_arg = matches[0].strip()
                if first_arg.startswith("'") or first_arg.startswith('"'):
                    return "Palimpzest SEM_ORDER_BY first argument must be bare column name, not quoted string"
            
            # Check for overly vague prompts
            sem_func_prompts = []
            for func in ['SEM_WHERE', 'SEM_SELECT']:
                if func in query_upper:
                    pattern = rf"{func}\s*\(['\"]([^'\"]+)['\"]"
                    prompts = re.findall(pattern, query, re.IGNORECASE)
                    sem_func_prompts.extend(prompts)
            
            for prompt in sem_func_prompts:
                # Check for very short/vague prompts (< 10 chars, too generic)
                if len(prompt.strip()) < 10:
                    logger.warning(
                        f"Palimpzest prompt very short ('{prompt}'). "
                        "Be specific about what to extract/check for better results."
                    )
                # Check for prompts without clear instructions
                if not any(word in prompt.lower() for word in ['extract', 'return', 'check', 'is', 'has', 'mentions', 'contains', 'true', 'false']):
                    logger.warning(
                        f"Palimpzest prompt lacks clear instruction ('{prompt[:50]}...'). "
                        "Include action words like 'Extract', 'Return true if', 'Check whether', etc."
                    )
        
        # Unified Backend Validation
        elif backend == 'unified':
            # Column-based functions must have column as first argument
            column_based_funcs = ['SEM_AGG', 'SEM_ORDER_BY', 'SEM_GROUP_BY']
            for func in column_based_funcs:
                if func in query_upper:
                    pattern = rf'{func}\s*\(\s*([^,\)]+)\s*,'
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    if not matches:
                        return f"Unified {func} must specify column as first argument"
                    
                    first_arg = matches[0].strip()
                    if first_arg.startswith("'") or first_arg.startswith('"'):
                        return f"Unified {func} first argument must be bare column name, not quoted string"
        
        # Check for queries without FROM clause (when data access is needed)
        if self.db_type == 'mysql':
            # Detect if query tries to access table data without FROM
            select_match = re.search(r'SELECT\s+(.+?)(?:;|$)', query, re.IGNORECASE | re.DOTALL)
            if select_match and 'FROM' not in query_upper:
                select_clause = select_match.group(1).strip()
                # Allow literal-only queries (e.g., SELECT 1, SELECT 'value')
                # But reject if it references columns or functions that need table data
                needs_from = False
                if re.search(r'\w+\.\w+', select_clause):  # table.column
                    needs_from = True
                elif not re.match(r'^[\d\s\'"\,\+\-\*\/\(\)]+$', select_clause.replace('AS', '').replace('as', '')):  # Not just literals
                    # Check if it's calling functions without table context
                    if 'DATEDIFF' in select_clause or 'TIMESTAMPDIFF' in select_clause or 'COUNT' in select_clause:
                        needs_from = True
                
                if needs_from:
                    return "Query must have a FROM clause"
        
        # ===== STAGE 5: Error Indicators and Malformed Content ====="
        
        # Check for error indicators
        error_indicators = ['UNDEFINED', 'ERROR', 'INVALID', 'NULL AS NULL', 'FIXME', 'TODO', '???']
        for indicator in error_indicators:
            if indicator in query_upper:
                return f"Query contains error indicator: '{indicator}'"
        
        # Check for incomplete template substitutions
        if '${' in query or ('{' in query and '}' not in query):
            return "Query contains incomplete template substitutions"
        
        # Check for Python-style placeholders in non-semantic function contexts
        # These would cause STRUCT() errors in SQLAlchemy
        # Look for {variable} in WHERE/AND/OR clauses outside of SEM_ function strings
        where_clause_match = re.search(r'WHERE\s+(.+?)(?:;|$|\s+(?:GROUP|ORDER|LIMIT|HAVING))', query, re.IGNORECASE | re.DOTALL)
        if where_clause_match:
            where_content = where_clause_match.group(1)
            # Remove semantic function content to avoid false positives
            cleaned = re.sub(r"SEM_\w+\s*\([^)]+\)", "", where_content, flags=re.IGNORECASE)
            # Check if there are placeholders outside semantic functions
            if re.search(r'\{[a-zA-Z_]\w*\}', cleaned):
                return "Query contains unsubstituted placeholders like {No} in WHERE clause. Use concrete column names or values."
        
        # All validation passed
        return None
    
    def _fix_mysql_group_by_issues(self, query: str) -> str:
        """
        Auto-fix MySQL ONLY_FULL_GROUP_BY violations.
        
        Common patterns:
        1. SELECT col1, col2, COUNT(*) GROUP BY col1  -> Add col2 to GROUP BY
        2. SELECT col1, col2, AGG() FROM ... (no GROUP BY) -> Wrap col1,col2 in ANY_VALUE()
        3. ORDER BY col NOT IN GROUP BY -> Add col to GROUP BY or remove ORDER BY
        4. Fix nested ANY_VALUE() issues with function calls like STR_TO_DATE
        
        Args:
            query: SQL query
            
        Returns:
            Fixed query
        """
        if self.db_type != 'mysql':
            return query
        
        query_upper = query.upper()
        
        # Pattern 1: Aggregates without GROUP BY
        # If there's COUNT/SUM/AVG/MAX/MIN but no GROUP BY, wrap non-aggregate columns in ANY_VALUE()
        has_agg = bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', query_upper))
        has_group_by = 'GROUP BY' in query_upper
        
        if has_agg and not has_group_by:
            # Extract SELECT columns
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_content = select_match.group(1)
                # Parse columns (simple heuristic)
                columns = [c.strip() for c in select_content.split(',')]
                
                fixed_columns = []
                for col in columns:
                    col_upper = col.upper()
                    # If column contains aggregate, keep as is
                    if any(agg in col_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'SEM_AGG(', 'SEM_SELECT(']):
                        fixed_columns.append(col)
                    # Check for date/time functions that should not be wrapped
                    elif any(func in col_upper for func in ['YEAR(', 'MONTH(', 'DAY(', 'DATE(', 'STR_TO_DATE(']):
                        # These functions should use their arguments from GROUP BY, not be wrapped
                        fixed_columns.append(col)
                    # If column is simple identifier (not function call), wrap in ANY_VALUE
                    elif re.match(r'^[a-zA-Z_`][\w`]*(\s+AS\s+\w+)?$', col.strip(), re.IGNORECASE):
                        col_name = col.split('AS')[0].strip() if 'AS' in col.upper() else col.strip()
                        alias = col.split('AS')[1].strip() if 'AS' in col.upper() else ''
                        wrapped = f"ANY_VALUE({col_name})"
                        if alias:
                            wrapped += f" AS {alias}"
                        fixed_columns.append(wrapped)
                        logger.info(f"Wrapped {col} in ANY_VALUE() to fix ONLY_FULL_GROUP_BY")
                    else:
                        fixed_columns.append(col)
                
                if fixed_columns != columns:
                    new_select = ', '.join(fixed_columns)
                    query = re.sub(
                        r'SELECT\s+(.*?)\s+FROM',
                        f'SELECT {new_select} FROM',
                        query,
                        count=1,
                        flags=re.IGNORECASE | re.DOTALL
                    )
        
        # Fix ORDER BY with aggregate functions that don't match GROUP BY
        if has_group_by:
            order_by_match = re.search(r'ORDER BY\s+(.*?)(?:\s+LIMIT|\s+$)', query, re.IGNORECASE)
            if order_by_match:
                order_clause = order_by_match.group(1)
                # If ORDER BY uses ANY_VALUE() and there's aggregate, it might need fixing
                if 'ANY_VALUE(' in order_clause.upper():
                    # Remove ANY_VALUE from ORDER BY if aggregate is already applied
                    fixed_order = re.sub(r'ANY_VALUE\((.*?)\)', r'\1', order_clause, flags=re.IGNORECASE)
                    query = query.replace(order_clause, fixed_order)
                    logger.info(f"Removed ANY_VALUE from ORDER BY clause")
        
        return query
    
    def _fix_scalar_subquery(self, query: str) -> str:
        """
        Auto-fix scalar subqueries that might return multiple rows.
        
        Patterns:
        - WHERE col = (SELECT ...) -> Add LIMIT 1 to subquery
        - SELECT (SELECT ...) AS alias -> Add LIMIT 1 to subquery
        
        Args:
            query: SQL query
            
        Returns:
            Fixed query
        """
        # Find scalar subqueries (parenthesized SELECT not preceded by IN/EXISTS/ANY/ALL)
        # Pattern: = (SELECT ...) or IN (SELECT ...)
        
        def add_limit_to_subquery(match):
            """Add LIMIT 1 to subquery if not already present."""
            subquery = match.group(0)
            subquery_upper = subquery.upper()
            
            # Don't modify if already has LIMIT or if it's used with IN/EXISTS
            if 'LIMIT' in subquery_upper:
                return subquery
            
            # Find the closing parenthesis
            depth = 0
            for i, char in enumerate(subquery):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        # Insert LIMIT 1 before the closing paren
                        fixed = subquery[:i] + ' LIMIT 1' + subquery[i:]
                        logger.info(f"Added LIMIT 1 to scalar subquery to prevent multi-row error")
                        return fixed
            
            return subquery
        
        # Pattern: = (SELECT ...) or column = (SELECT ...)
        query = re.sub(
            r'=\s*\(\s*SELECT\s+.*?\)',
            add_limit_to_subquery,
            query,
            flags=re.IGNORECASE
        )
        
        return query
    
    def _normalize_column_quotes(self, query: str) -> str:
        """
        Normalize column name quotes based on database type.
        - DuckDB/SQLite: backticks (`) -> double quotes (")
        - MySQL: double quotes (") -> backticks (`)
        
        Args:
            query: SQL query that may contain quoted column names
            
        Returns:
            Query with normalized column name quotes
        """
        if self.db_type == 'mysql':
            target_quote = '`'
            source_quote = '"'
        else:  # duckdb or sqlite
            target_quote = '"'
            source_quote = '`'
        
        result = []
        in_single_quote = False
        i = 0
        
        while i < len(query):
            char = query[i]
            
            if char == "'" and (i == 0 or query[i-1] != '\\'):
                in_single_quote = not in_single_quote
                result.append(char)
            elif char == source_quote and not in_single_quote:
                result.append(target_quote)
            else:
                result.append(char)
            
            i += 1
        
        normalized = ''.join(result)
        
        if source_quote in query:
            logger.info(f"Normalized {source_quote} to {target_quote} for {self.db_type}")
            logger.debug(f"Original: {query[:100]}...")
            logger.debug(f"Normalized: {normalized[:100]}...")
        
        return normalized
    
    def _get_semantic_function_syntax(self, backend: str) -> str:
        """
        Get semantic function syntax specification for the given backend.
        
        Args:
            backend: Target backend ('lotus', 'docetl', 'palimpzest', 'unified')
            
        Returns:
            Formatted syntax specification string
        """
        if backend == 'lotus':
            return """LOTUS Semantic Function Syntax (MUST follow exactly):
- SEM_WHERE('prompt with {column} references', 'lotus')
  Example: SEM_WHERE('{bio} mentions machine learning', 'lotus')

- SEM_SELECT('prompt with {column} references', 'lotus') AS alias
  Example: SEM_SELECT('Extract the first name from {manager} as a plain text without explanation', 'lotus') AS extracted_name

- SEM_AGG('prompt with {column} references', 'lotus') AS alias
  Example: SEM_AGG('Summarize {notes} in a single sentence', 'lotus') AS summary

- SEM_ORDER_BY('prompt with {column} references', 'lotus')
  Example: SEM_ORDER_BY('Rank {title} by relevance to Sci-Fi genre', 'lotus')

- SEM_GROUP_BY(column_name, num_groups, 'lotus')
  Example: SEM_GROUP_BY(category, 3, 'lotus')

- SEM_JOIN(table1, table2, 'prompt with {col:left} and {col:right}', 'lotus')
  Example: SEM_JOIN(products, reviews, '{name:left} is mentioned in {product:right}', 'lotus')

* Reference columns in templates using {column_name} syntax for single table queries
* CRITICAL: When query has Table ALIASES or JOIN clauses, ALWAYS use {table_alias.column_name} or {table.column_name} syntax in semantic functions"""
        
        elif backend == 'docetl':
            return """DocETL Semantic Function Syntax (MUST follow exactly):

- SEM_WHERE('prompt with {{ input.column }} Jinja2 syntax', 'docetl')
  Example: SEM_WHERE('Biography: {{ input.bio }}\n\nReturn true if mentions ML, otherwise false', 'docetl')

- SEM_SELECT('prompt with {{ input.column }} Jinja2 syntax', 'docetl') AS alias
  Example: SEM_SELECT('Manager: {{ input.manager }}\n\nExtract the first name as a plain text without explanation', 'docetl') AS name

- SEM_AGG 
--- [column-specified]:
    SEM_AGG(column_name, 'natural language prompt', 'docetl') AS alias
    Example: SEM_AGG(bio, 'Summarize all bios in two sentences', 'docetl') AS summary
--- [global]:
    SEM_AGG('natural language prompt', 'docetl') AS alias
    Example: SEM_AGG('Summarize all notes in two or three sentences', 'docetl') AS summary

- SEM_ORDER_BY(column_name, 'natural language prompt', 'docetl')
  Example: SEM_ORDER_BY(summary, 'Order by relevance to Sci-Fi genre', 'docetl')

- SEM_GROUP_BY(column_name, num_groups, 'docetl')
  Example: SEM_GROUP_BY(category, 3, 'docetl')

- SEM_JOIN(table1, table2, 'prompt with {{ left.col }} and {{ right.col }}', 'docetl')
  Example: SEM_JOIN(products, reviews, '{{ left.name }} is mentioned in {{ right.product }}', 'docetl')
  
* For single table queries: {{ input.column_name }}
* CRITICAL: When query has Table ALIASES or JOIN clauses, ALWAYS use {{ input.table_alias.column_name }} or {{ input.table.column_name }} syntax in semantic functions"""
        
        elif backend == 'palimpzest':
            return """Palimpzest Semantic Function Syntax (MUST follow exactly):
- SEM_WHERE('natural language description', 'palimpzest')
  Example: SEM_WHERE('This person has machine learning experience', 'palimpzest')
  Note: NO column references needed, Palimpzest auto-accesses row data

- SEM_SELECT('natural language instruction', 'palimpzest') AS alias
  Example: SEM_SELECT('Extract the first name from manager as a plain text without explanation', 'palimpzest') AS name

- SEM_ORDER_BY(column_name, 'natural language criteria', 'palimpzest')
  Example: SEM_ORDER_BY(description, 'Order by relevance to Sci-Fi genre', 'palimpzest')
  Note: MUST specify column as first argument

- SEM_JOIN(table1, table2, 'natural language matching criteria', 'palimpzest')
  Example: SEM_JOIN(products, reviews, 'The product is mentioned in the review', 'palimpzest')"""
        
        elif backend == 'unified':
            return """Unified Semantic Function Syntax (MUST follow exactly):
- SEM_WHERE('natural language description')
  Example: SEM_WHERE('This person has machine learning experience')
  Note: No backend parameter, no column references needed

- SEM_SELECT('natural language instruction') AS alias
  Example: SEM_SELECT('Extract the person name') AS name

- SEM_AGG(column_name, 'natural language prompt') AS alias
  Example: SEM_AGG(bio, 'Summarize all bios') AS summary

- SEM_ORDER_BY(column_name, 'natural language criteria')
  Example: SEM_ORDER_BY(description, 'Order by relevance to topic')
  Note: MUST specify column as first argument

- SEM_GROUP_BY(column_name, num_groups)
  Example: SEM_GROUP_BY(category, 3)

- SEM_JOIN(table1, table2, 'natural language matching criteria')
  Example: SEM_JOIN(products, reviews, 'The review is about this product')"""
        
        return ""
    
    def _verify_and_optimize_query(
        self,
        query: str,
        question: str,
        schema: str,
        backend: str,
        temperature: float = 0.0
    ) -> str:
        """
        Use LLM to verify and optimize the generated SABER SQL query.
        
        Args:
            query: Generated SABER SQL query
            question: Original natural language question
            schema: Database schema
            backend: Target backend
            temperature: LLM temperature (default: 0.0 for deterministic output)
            
        Returns:
            Verified and potentially optimized SABER SQL query
        """
        db_constraints = self._get_db_constraints()
        
        # Get backend-specific semantic function syntax
        semantic_syntax = self._get_semantic_function_syntax(backend)
        
        verification_prompt = f"""You are an expert SABER SQL query optimizer.

SABER SQL is an extension of {self.db_type} SQL with semantic functions that process unstructured data semantically.

Backend: {backend}

{semantic_syntax}

{db_constraints}

Schema:
{schema}

Question: {question}

Generated SABER SQL Query:
{query}

Verification Checklist:
1. Correctness: Does it answer the question?
2. Syntax: Valid SABER SQL with no errors?
3. Database Compatibility: Uses only {self.db_type}-compatible SQL functions?
4. Semantic Functions: Follows EXACT syntax specification for '{backend}' backend shown above?
5. Optimization: Can it be improved for correctness or critical syntax errors?
   - PRESERVE semantic functions by default - they provide semantic understanding beyond simple pattern matching
   - ONLY replace semantic functions with SQL pattern matching (LIKE/LOCATE) if:
     * The question explicitly asks for exact substring/keyword matching (e.g., "contains the word X")
     * The semantic operation adds no value (e.g., extracting a column that already exists)
   - KEEP semantic functions for:
     * Understanding concepts, intent, or meaning (e.g., "positive sentiment", "relevant to topic")
     * Extracting or transforming unstructured text
     * Matching on semantics rather than exact keywords (e.g., "mentions ML" should use SEM_WHERE, not LOCATE)

CRITICAL Rules:
- If modifying semantic functions, STRICTLY follow the syntax specification for '{backend}' backend
- DO NOT change semantic function syntax to other formats
- DO NOT invent new semantic function parameters or argument orders
- Preserve exact template syntax (e.g., {{column}} for lotus, {{{{ input.column }}}} for docetl)

Instructions:
- If the query is already correct and optimal, output it exactly as-is
- If improvements are needed, output the corrected/optimized version
- Do NOT add explanations, comments, markdown, or code blocks
- Output ONLY the SABER SQL query itself

Output:"""
        
        try:
            logger.debug(f"Running LLM verification for query: {query[:100]}...")
            
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=temperature,
                max_tokens=1024,
                api_key=self.api_key,
                api_base=self.api_base
            )
            
            if not response or not response.choices or not response.choices[0].message.content:
                logger.warning("Empty verification response, using original query")
                return query
            
            result = response.choices[0].message.content.strip()
            result = self._clean_query_output(result)
            
            # Validate the result
            validation_error = self._verify_query(result, backend)
            if validation_error:
                logger.warning(f"LLM output failed validation: {validation_error}. Using original query.")
                return query
            
            # Check if query was changed
            if result.strip() != query.strip():
                logger.info(f"Query optimized by LLM:\nOriginal:  {query}\nOptimized: {result}")
            else:
                logger.info("Query verified by LLM (no changes)")
            
            return result
        
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}. Using original query.")
            return query
    
    def _clean_query_output(self, query: str) -> str:
        """
        Clean LLM output to extract pure SQL query.
        
        Args:
            query: Raw LLM output
            
        Returns:
            Cleaned SQL query
        """
        # Remove common code block markers
        if query.startswith("```sql"):
            query = query[6:]
        if query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
        
        return query.strip()

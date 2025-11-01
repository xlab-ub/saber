"""
SABER Query Generator - Generate SABER SQL from natural language.
"""
from typing import Dict, List, Optional
import logging
from textwrap import dedent

import litellm

logger = logging.getLogger(__name__)


class SABERQueryGenerator:
    """Generate SABER SQL queries from natural language questions."""
    
    DEFAULT_EXAMPLES = {
        'lotus': [
            {
                "schema": """
Table: employees
Columns: id (INTEGER), name (TEXT), department (TEXT), salary (INTEGER), bio (TEXT)
Sample rows:
  1, 'Alice Chen', 'Engineering', 95000, 'Alice is a senior engineer with expertise in distributed systems...'
  2, 'Bob Smith', 'Sales', 75000, 'Bob has been with the company for 5 years selling enterprise solutions...'
  3, 'Carol Lee', 'Engineering', 88000, 'Carol specializes in machine learning and data pipelines...'
""",
                "question": "Which engineering employees have experience in machine learning?",
                "saber_sql": """
SELECT name, bio 
FROM employees 
WHERE department = 'Engineering' 
  AND SEM_WHERE('{bio} mentions machine learning experience', 'lotus')
"""
            },
            {
                "schema": """
Table: products
Columns: id (INTEGER), name (TEXT), category (TEXT), price (DECIMAL), description (TEXT)
Sample rows:
  1, 'Laptop Pro', 'Electronics', 1299.99, 'High-performance laptop with 16GB RAM...'
  2, 'Office Chair', 'Furniture', 299.99, 'Ergonomic chair with lumbar support...'
  3, 'Wireless Mouse', 'Electronics', 49.99, 'Bluetooth mouse with precision tracking...'
""",
                "question": "What are the top 3 most expensive products with positive descriptions?",
                "saber_sql": """
SELECT name, price, 
       SEM_SELECT('Summarize the customer {description} without explanation.', 'lotus') AS summary
FROM products
WHERE SEM_WHERE('{description} has positive tone', 'lotus')
ORDER BY price DESC
LIMIT 3
"""
            },
            {
                "schema": """
Table: restaurants
Columns: id (INTEGER), name (TEXT), cuisine (TEXT), rating (DECIMAL), location (TEXT)
Table: reviews
Columns: id (INTEGER), restaurant_id (INTEGER), reviewer (TEXT), comment (TEXT), score (INTEGER)
Sample rows (restaurants):
  1, 'La Bella Italia', 'Italian', 4.5, 'Downtown'
  2, 'Sushi Master', 'Japanese', 4.8, 'Uptown'
Sample rows (reviews):
  1, 1, 'John', 'Amazing pasta and great service!', 5
  2, 1, 'Mary', 'Authentic Italian flavors', 4
""",
                "question": "Find restaurants with reviews mentioning authentic cuisine",
                "saber_sql": """
SELECT r.name, r.cuisine, rv.comment
FROM restaurants AS r
JOIN reviews AS rv ON r.id = rv.restaurant_id
WHERE SEM_WHERE('{reviews.comment} mentions authentic or traditional cuisine', 'lotus')
"""
            },
            {
                "schema": """
Table: drivers
Columns: id (INTEGER), name (TEXT), position (INTEGER), team (TEXT)
Table: documents
Columns: id (INTEGER), title (TEXT), content (TEXT)
Sample rows (drivers):
  1, 'Lewis Hamilton', 4, 'Mercedes'
  2, 'Max Verstappen', 1, 'Red Bull'
Sample rows (documents):
  1, 'Lewis Hamilton', 'British racing driver born in Stevenage, England...'
  2, 'Max Verstappen', 'Dutch-Belgian racing driver from Hasselt, Belgium...'
""",
                "question": "Get driver and nationality for drivers in position 4",
                "saber_sql": """
SELECT d.name AS Driver,
       SEM_SELECT('Extract nationality from {documents.content} without explanation', 'lotus') AS nationality
FROM drivers AS d
JOIN documents ON documents.title = d.name
WHERE d.position = 4
"""
            },
            {
                "schema": """
Table: fruits
Columns: id (INTEGER), name (TEXT), price (DECIMAL), category (TEXT)
Sample rows:
  1, 'Golden Delicious', 12.5, 'fresh'
  2, 'Red Delicious', 11.0, 'fresh'
  3, 'Banana', 8.0, 'fresh'
""",
                "question": "Group similar fruits and find average price per group",
                "saber_sql": """
SELECT SEM_GROUP_BY(name, 2, 'lotus') AS fruit_group, 
       AVG(price) AS avg_price
FROM fruits
GROUP BY SEM_GROUP_BY(name, 2, 'lotus')
ORDER BY avg_price DESC
"""
            }
        ],
        'docetl': [
            {
                "schema": """
Table: employees
Columns: id (INTEGER), name (TEXT), department (TEXT), salary (INTEGER), bio (TEXT)
Sample rows:
  1, 'Alice Chen', 'Engineering', 95000, 'Alice is a senior engineer with expertise in distributed systems...'
  2, 'Bob Smith', 'Sales', 75000, 'Bob has been with the company for 5 years selling enterprise solutions...'
  3, 'Carol Lee', 'Engineering', 88000, 'Carol specializes in machine learning and data pipelines...'
""",
                "question": "Which engineering employees have experience in machine learning?",
                "saber_sql": """
SELECT name, bio 
FROM employees 
WHERE department = 'Engineering' 
  AND SEM_WHERE('Bio: {{ input.bio }}
  
Return true if this person has machine learning experience, otherwise false.', 'docetl')
"""
            },
            {
                "schema": """
Table: products
Columns: id (INTEGER), name (TEXT), category (TEXT), price (DECIMAL), description (TEXT)
Sample rows:
  1, 'Laptop Pro', 'Electronics', 1299.99, 'High-performance laptop with 16GB RAM...'
  2, 'Office Chair', 'Furniture', 299.99, 'Ergonomic chair with lumbar support...'
  3, 'Wireless Mouse', 'Electronics', 49.99, 'Bluetooth mouse with precision tracking...'
""",
                "question": "Extract sentiment from product descriptions",
                "saber_sql": """
SELECT name, price,
       SEM_SELECT('{{ input.description }}

Extract the customer sentiment (positive/negative/neutral) without explanation.', 'docetl') AS sentiment
FROM products
WHERE SEM_WHERE('Description: {{ input.description }}

Return true if the description has positive tone, otherwise false.', 'docetl')
ORDER BY price DESC
"""
            },
            {
                "schema": """
Table: restaurants
Columns: id (INTEGER), name (TEXT), cuisine (TEXT), rating (DECIMAL), location (TEXT)
Table: chefs
Columns: id (INTEGER), restaurant_id (INTEGER), name (TEXT), specialty (TEXT), experience (TEXT)
Sample rows (restaurants):
  1, 'La Bella Italia', 'Italian', 4.5, 'Downtown'
  2, 'Sushi Master', 'Japanese', 4.8, 'Uptown'
Sample rows (chefs):
  1, 1, 'Marco Rossi', 'Pasta', 'Trained in Rome, 15 years experience...'
  2, 2, 'Takeshi Yamamoto', 'Sushi', 'Tokyo culinary school, master sushi chef...'
""",
                "question": "Find restaurants where the chef has traditional training",
                "saber_sql": """
SELECT r.name, r.cuisine, c.name AS chef_name, c.experience
FROM restaurants AS r
JOIN chefs AS c ON r.id = c.restaurant_id
WHERE SEM_WHERE('Chef Experience: {{ input.chefs_experience }}

Return true if the chef has traditional or classical culinary training, otherwise false.', 'docetl')
"""
            },
            {
                "schema": """
Table: drivers
Columns: id (INTEGER), name (TEXT), position (INTEGER), team (TEXT)
Table: documents
Columns: id (INTEGER), title (TEXT), content (TEXT)
Sample rows (drivers):
  1, 'Lewis Hamilton', 4, 'Mercedes'
  2, 'Max Verstappen', 1, 'Red Bull'
Sample rows (documents):
  1, 'Lewis Hamilton', 'British racing driver born in Stevenage, England...'
  2, 'Max Verstappen', 'Dutch-Belgian racing driver from Hasselt, Belgium...'
""",
                "question": "Get driver and nationality for drivers in position 4",
                "saber_sql": """
SELECT d.name AS Driver,
       SEM_SELECT('{{ input.documents_content }}

Extract the nationality without explanation.', 'docetl') AS nationality
FROM drivers AS d
JOIN documents ON documents.title = d.name
WHERE d.position = 4
"""
            },
            {
                "schema": """
Table: papers
Columns: id (INTEGER), title (TEXT), authors (TEXT), year (INTEGER), abstract (TEXT)
Sample rows:
  1, 'Attention Is All You Need', 'Vaswani et al.', 2017, 'We propose a new architecture...'
  2, 'BERT: Pre-training', 'Devlin et al.', 2019, 'We introduce BERT...'
""",
                "question": "Summarize research contributions by year",
                "saber_sql": """
SELECT year,
       SEM_AGG(abstract, 'Summarize the main contributions in one sentence.', 'docetl') AS summary
FROM papers
GROUP BY year
ORDER BY year DESC
"""
            }
        ],
        'palimpzest': [
            {
                "schema": """
Table: employees
Columns: id (INTEGER), name (TEXT), department (TEXT), salary (INTEGER), bio (TEXT)
Sample rows:
  1, 'Alice Chen', 'Engineering', 95000, 'Alice is a senior engineer with expertise in distributed systems...'
  2, 'Bob Smith', 'Sales', 75000, 'Bob has been with the company for 5 years selling enterprise solutions...'
  3, 'Carol Lee', 'Engineering', 88000, 'Carol specializes in machine learning and data pipelines...'
""",
                "question": "Which engineering employees have experience in machine learning?",
                "saber_sql": """
SELECT name, bio 
FROM employees 
WHERE department = 'Engineering' 
  AND SEM_WHERE('This person has machine learning experience', 'palimpzest')
"""
            },
            {
                "schema": """
Table: products
Columns: id (INTEGER), name (TEXT), category (TEXT), price (DECIMAL), description (TEXT)
Sample rows:
  1, 'Laptop Pro', 'Electronics', 1299.99, 'High-performance laptop with 16GB RAM...'
  2, 'Office Chair', 'Furniture', 299.99, 'Ergonomic chair with lumbar support...'
  3, 'Wireless Mouse', 'Electronics', 49.99, 'Bluetooth mouse with precision tracking...'
""",
                "question": "What are the top 3 most expensive products with positive descriptions?",
                "saber_sql": """
SELECT name, price, 
       SEM_SELECT('Summarize the product description', 'palimpzest') AS summary
FROM products
WHERE SEM_WHERE('The description has a positive tone', 'palimpzest')
ORDER BY price DESC
LIMIT 3
"""
            },
            {
                "schema": """
Table: restaurants
Columns: id (INTEGER), name (TEXT), cuisine (TEXT), rating (DECIMAL), location (TEXT)
Table: reviews
Columns: id (INTEGER), restaurant_id (INTEGER), reviewer (TEXT), comment (TEXT), score (INTEGER)
Sample rows (restaurants):
  1, 'La Bella Italia', 'Italian', 4.5, 'Downtown'
  2, 'Sushi Master', 'Japanese', 4.8, 'Uptown'
Sample rows (reviews):
  1, 1, 'John', 'Amazing pasta and great service!', 5
  2, 1, 'Mary', 'Authentic Italian flavors', 4
""",
                "question": "Find restaurants with reviews mentioning authentic cuisine",
                "saber_sql": """
SELECT r.name, r.cuisine, rv.comment
FROM restaurants AS r
JOIN reviews AS rv ON r.id = rv.restaurant_id
WHERE SEM_WHERE('The review mentions authentic or traditional cuisine', 'palimpzest')
"""
            },
            {
                "schema": """
Table: products
Columns: id (INTEGER), name (TEXT), price (DECIMAL), description (TEXT)
Sample rows:
  1, 'Laptop Pro', 1299.99, 'High-performance laptop for professionals'
  2, 'Gaming Mouse', 79.99, 'RGB mouse with precision tracking'
  3, 'Webcam HD', 89.99, 'Crystal clear video for meetings'
""",
                "question": "Order products by relevance to remote work",
                "saber_sql": """
SELECT name, price, description
FROM products
ORDER BY SEM_ORDER_BY(description, 'Order by relevance to remote work and home office setup', 'palimpzest')
LIMIT 5
"""
            },
            {
                "schema": """
Table: fruits
Columns: id (INTEGER), name (TEXT), price (DECIMAL), note (TEXT)
Sample rows:
  1, 'Apple', 12.5, 'Fresh from local farms'
  2, 'Banana', 8.0, 'Good for smoothies'
  3, 'Orange', 10.0, 'High vitamin C content'
""",
                "question": "Rank fruits by how healthy they are based on descriptions",
                "saber_sql": """
SELECT name, price, note
FROM fruits
ORDER BY SEM_ORDER_BY(note, 'Order by health benefits', 'palimpzest')
"""
            }
        ],
    }
    
    SYSTEM_PROMPTS = {        
        'lotus': dedent("""
        Generate SABER SQL with LOTUS backend given a database schema and natural language question.
        
        SABER SQL is a superset of standard SQL on DuckDB with semantic functions. All semantic functions must specify 'lotus' backend.
        
        LOTUS Semantic Functions:
        
        1. SEM_WHERE('condition', 'lotus') - Semantic filtering
           Template: Use {column_name} to reference columns in condition
           Example: WHERE SEM_WHERE('{bio} mentions machine learning', 'lotus')
           With JOIN: Use {table.column_name} syntax
           Example: WHERE SEM_WHERE('{documents.content} mentions AI', 'lotus')
        
        2. SEM_SELECT('description', 'lotus') AS alias - Semantic extraction
           Template: Use {column_name} in description
           Example: SELECT SEM_SELECT('Extract main point from {abstract} without explanation', 'lotus') AS main_point
           With JOIN: Use {table.column_name} syntax
           Example: SELECT SEM_SELECT('Extract nationality from {documents.content} without explanation', 'lotus') AS nationality
        
        3. SEM_AGG('description', 'lotus') AS alias - Semantic aggregation
           Template: Reference columns with {column_name}
           Example: SELECT SEM_AGG('Summarize {note} values', 'lotus') AS summary
        
        4. SEM_ORDER_BY('criteria', 'lotus') - Semantic ordering
           Template: Use {column_name} for sorting criteria
           Example: ORDER BY SEM_ORDER_BY('Which {name} is most relevant?', 'lotus')
        
        5. SEM_GROUP_BY(column, n_groups, 'lotus') - Semantic grouping
           Groups similar values into n_groups clusters
           Example: GROUP BY SEM_GROUP_BY(name, 3, 'lotus')
        
        6. SEM_JOIN(table1, table2, 'condition', 'lotus') - Semantic join
           Template: Use {column:left} and {column:right} to reference columns
           Example: FROM SEM_JOIN(products, categories, '{name:left} belongs to {type:right}', 'lotus')
        
        Guidelines:
        - Use standard SQL for structured operations (numeric filters, exact matches, ID joins)
        - Use semantic functions ONLY when text understanding is required
        - CRITICAL: For extracting data from text, ALWAYS use SEM_SELECT, NOT string functions like SUBSTRING_INDEX or REGEXP_EXTRACT
        - The underlying database is DuckDB - do not use MySQL/PostgreSQL-specific functions
        - Always specify 'lotus' as the backend parameter
        - Reference columns in templates using {column_name} syntax for single table queries
        - CRITICAL: When query has JOIN clauses, ALWAYS use {table.column_name} syntax in semantic functions
        - For column names with spaces or special characters, use double quotes: "Column Name"
        - Never use backticks (`) for quoting - always use double quotes (")
        - When grouping by extracted text values, use the same SEM_SELECT in both SELECT and GROUP BY clauses
        
        Output only the SABER SQL query, no explanations.
        """).strip(),
        
        'docetl': dedent("""
        Generate SABER SQL with DocETL backend given a database schema and natural language question.
        
        SABER SQL is a superset of standard SQL on DuckDB with semantic functions. All semantic functions must specify 'docetl' backend.
        
        DocETL Semantic Functions (use Jinja2 templates):
        
        1. SEM_WHERE('condition', 'docetl') - Semantic filtering
           Template: Use {{ input.column_name }} to access row values (single table)
           Example: WHERE SEM_WHERE('Bio: {{ input.bio }}\\n\\nReturn true if mentions ML, else false.', 'docetl')
           With JOIN: Use {{ input.table_column_name }} with table prefix
           Example: WHERE SEM_WHERE('Content: {{ input.documents_content }}\\n\\nReturn true if mentions AI.', 'docetl')
        
        2. SEM_SELECT('prompt', 'docetl') AS alias - Semantic extraction
           Template: Access row data with {{ input.column_name }} (single table)
           Example: SELECT SEM_SELECT('{{ input.abstract }}\\n\\nExtract the main contribution without explanation.', 'docetl') AS contribution
           With JOIN: Use {{ input.table_column_name }} with table prefix
           Example: SELECT SEM_SELECT('{{ input.documents_content }}\\n\\nExtract nationality without explanation.', 'docetl') AS nationality
        
        3. SEM_AGG(column, 'prompt', 'docetl') AS alias - Semantic aggregation with column
           OR SEM_AGG('prompt', 'docetl') AS alias - Global aggregation
           Template: For column-based, prompt processes column values. For global, prompt sees all data.
           Example: SELECT SEM_AGG(note, 'Summarize all notes in one sentence.', 'docetl') AS summary
           Example: SELECT SEM_AGG('What are common characteristics?', 'docetl') AS summary
        
        4. SEM_ORDER_BY(column, 'criteria', 'docetl') - Semantic ordering
           Template: Prompt describes how to order the specified column
           Example: ORDER BY SEM_ORDER_BY(name, 'Order by relevance to winter fruits', 'docetl')
        
        5. SEM_GROUP_BY(column, n_groups, 'docetl') - Semantic grouping
           Groups similar values into n_groups clusters
           Example: GROUP BY SEM_GROUP_BY(name, 3, 'docetl')
        
        6. SEM_JOIN(table1, table2, 'condition', 'docetl') - Semantic join
           Template: Use {{ left.column }} and {{ right.column }} to access values
           Example: FROM SEM_JOIN(products, reviews, 'Product: {{ left.name }}\\nReview: {{ right.text }}\\n\\nDoes review match product?', 'docetl')
        
        Guidelines:
        - Use Jinja2 syntax: {{ input.column_name }} for current row, {{ left.col }}/{{ right.col }} for joins
        - Use standard SQL for structured operations
        - CRITICAL: For extracting data from text, ALWAYS use SEM_SELECT, NOT string functions like SUBSTRING_INDEX or REGEXP_EXTRACT
        - The underlying database is DuckDB - do not use MySQL/PostgreSQL-specific functions
        - Always specify 'docetl' as the backend parameter
        - For single table queries: {{ input.column_name }}
        - CRITICAL: When query has JOIN clauses, use {{ input.table_column_name }} format (underscore, not dot)
        - For SEM_JOIN operation: use {{ left.column }} and {{ right.column }}
        - For column names with spaces or special characters, use double quotes: "Column Name"
        - Never use backticks (`) for quoting - always use double quotes (")
        - When grouping by extracted text values, use the same SEM_SELECT in both SELECT and GROUP BY clauses
        
        Output only the SABER SQL query, no explanations.
        """).strip(),

        'palimpzest': dedent("""
        Generate SABER SQL with Palimpzest backend given a database schema and natural language question.
        
        SABER SQL is a superset of standard SQL on DuckDB with semantic functions. All semantic functions must specify 'palimpzest' backend.
        
        Palimpzest Semantic Functions (use pure natural language):
        
        1. SEM_WHERE('instruction', 'palimpzest') - Semantic filtering
           Use plain natural language instructions, no templates
           Example: WHERE SEM_WHERE('This person has machine learning experience', 'palimpzest')
        
        2. SEM_SELECT('instruction', 'palimpzest') AS alias - Semantic extraction
           Use plain natural language instructions
           Example: SELECT SEM_SELECT('Extract the main contribution from the abstract without explanation', 'palimpzest') AS contribution
        
        3. SEM_ORDER_BY(column, 'instruction', 'palimpzest') - Semantic ordering
           Must specify column to sort by, use plain natural language for criteria
           Example: ORDER BY SEM_ORDER_BY(description, 'Order by relevance to remote work', 'palimpzest')
        
        4. SEM_JOIN(table1, table2, 'instruction', 'palimpzest') - Semantic join
           Use plain natural language for matching criteria
           Example: FROM SEM_JOIN(products, reviews, 'The review matches this product', 'palimpzest')
        
        Guidelines:
        - Use standard SQL for structured operations (numeric filters, exact matches, ID joins)
        - Use semantic functions ONLY when text understanding is required
        - CRITICAL: For extracting data from text, ALWAYS use SEM_SELECT, NOT string functions like SUBSTRING_INDEX or REGEXP_EXTRACT
        - The underlying database is DuckDB - do not use MySQL/PostgreSQL-specific functions
        - Always specify 'palimpzest' as the backend parameter
        - Write instructions in plain natural language - NO templates, NO {column} syntax
        - Palimpzest automatically has access to all row data in the instruction context
        - Supported operations: SEM_WHERE, SEM_SELECT, SEM_ORDER_BY (with column), SEM_JOIN
        - NOT supported: SEM_AGG, SEM_GROUP_BY, SEM_DISTINCT (use standard SQL aggregation and GROUP BY)
        - CRITICAL: SEM_ORDER_BY MUST include column argument: SEM_ORDER_BY(column, 'instruction', 'palimpzest')
        - For column names with spaces or special characters, use double quotes: "Column Name"
        - Never use backticks (`) for quoting - always use double quotes (")
        - When grouping by extracted text values, use the same SEM_SELECT in both SELECT and GROUP BY clauses
        - Never use backticks (`) for quoting - always use double quotes (")
        
        Output only the SABER SQL query, no explanations.
        """).strip(),
    }
    
    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None, backend: str = 'lotus', 
                 max_retries: int = 3, fallback_enabled: bool = True):
        """
        Initialize query generator.
        
        Args:
            model: LiteLLM model identifier (e.g., 'openai/gpt-4o-mini', 'hosted_vllm/openai/gpt-oss-20b')
            api_key: API key (optional for local models)
            api_base: API base URL (for local models)
            backend: Target backend for generated queries ('lotus' or 'docetl')
            max_retries: Maximum number of retry attempts for failed queries (default: 3)
            fallback_enabled: Whether to enable fallback to simpler queries (default: True)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.backend = backend
        self.max_retries = max_retries
        self.fallback_enabled = fallback_enabled
        
        if backend not in ['lotus', 'docetl', 'palimpzest']:
            raise ValueError(f"Backend must be 'lotus', 'docetl', or 'palimpzest', got '{backend}'")
    
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
        schema += f"Sample rows ({min(max_rows, len(df))} of {len(df)} total):\n"
        for idx, row in df.head(max_rows).iterrows():
            values = ', '.join([repr(v) if isinstance(v, str) else str(v) for v in row.values])
            schema += f"  {values}\n"
        return schema
    
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
        max_tokens: int = 512
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
            
        Returns:
            SABER SQL query string
        """
        # Use specified backend or fall back to default
        backend = backend or self.backend
        if backend not in ['lotus', 'docetl', 'palimpzest']:
            raise ValueError(f"Backend must be 'lotus', 'docetl', or 'palimpzest', got '{backend}'")
        
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
        
        # Build prompt with backend-specific system message
        messages = [{"role": "system", "content": self.SYSTEM_PROMPTS[backend]}]
        
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
                    
                    feedback = f"ERROR: {str(e)}\n\n"
                    if 'SUBSTRING_INDEX' in str(e):
                        feedback += "SUBSTRING_INDEX is a MySQL function not supported in DuckDB. Use SEM_SELECT to extract text data:\n"
                        feedback += f"SELECT SEM_SELECT('Extract country code from {{column_name}}', '{backend}') AS country, COUNT(*) FROM table GROUP BY SEM_SELECT('Extract country code from {{column_name}}', '{backend}')"
                    elif 'REGEXP_EXTRACT' in str(e):
                        feedback += f"Do not use REGEXP_EXTRACT. Use SEM_SELECT for text extraction:\n"
                        feedback += f"SELECT SEM_SELECT('Extract needed data from {{column_name}}', '{backend}') AS result FROM table"
                    else:
                        feedback += f"Generate a valid SABER SQL query using semantic functions (SEM_SELECT, SEM_WHERE, etc.) with '{backend}' backend."
                    
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
WHERE SEM_WHERE('Content: {{{{ input.{first_text_col} }}}}\\n\\nReturn true if relevant to the question.', 'docetl')
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
    
    def _verify_query(self, query: str, backend: str) -> Optional[str]:
        """
        Verify that generated query has valid structure and syntax.
        
        Args:
            query: Generated SQL query
            backend: Target backend
            
        Returns:
            Error message if validation fails, None if valid
        """
        import re
        
        # Basic checks
        query_upper = query.upper()
        
        # 1. Must contain SELECT
        if 'SELECT' not in query_upper:
            return "Query must contain SELECT statement"
        
        # 2. Must contain FROM
        if 'FROM' not in query_upper:
            return "Query must contain FROM clause"
        
        # 3. Check for balanced parentheses (excluding those inside strings)
        # Parse character by character, tracking if we're inside a string
        paren_balance = 0
        in_single_quote = False
        in_double_quote = False
        i = 0
        
        while i < len(query):
            ch = query[i]
            
            # Handle escape sequences
            if ch == '\\' and i + 1 < len(query):
                i += 2  # Skip escaped character
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
        
        # 4. Check for balanced quotes
        if in_single_quote or in_double_quote:
            return "Unbalanced quotes"
        
        # 5. Check for unsupported SQL functions FIRST (before checking semantic functions)
        # These are common MySQL/PostgreSQL functions that don't exist in DuckDB
        unsupported_functions = [
            'SUBSTRING_INDEX',  # MySQL function for string splitting
            'REGEXP_EXTRACT',   # PostgreSQL-specific
            'STR_TO_DATE',      # MySQL date parsing
            'DATE_FORMAT',      # MySQL date formatting
            'CONCAT_WS',        # MySQL concat with separator (use || instead)
        ]
        
        for func in unsupported_functions:
            if func in query_upper:
                return f"Query uses unsupported function '{func}'. Use semantic functions (SEM_SELECT, SEM_WHERE, etc.) for text extraction instead of string functions."
        
        # 6. Verify semantic functions have correct backend parameter
        semantic_funcs = ['SEM_WHERE', 'SEM_SELECT', 'SEM_AGG', 'SEM_ORDER_BY', 'SEM_GROUP_BY', 'SEM_JOIN']
        
        for func in semantic_funcs:
            if func in query_upper:
                # Find all occurrences of this function
                pattern = rf'{func}\s*\([^)]*\)'
                matches = re.finditer(pattern, query, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    func_call = match.group(0)
                    
                    # Check if backend parameter is present
                    if f"'{backend}'" not in func_call and f'"{backend}"' not in func_call:
                        return f"{func} call missing or incorrect backend parameter: expected '{backend}'"
                    
                    # Verify function has at least 2 arguments (except SEM_GROUP_BY which has 3)
                    # Count commas outside of quotes
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
                    
                    # Most functions need at least 2 args (arg + backend)
                    min_commas = 1
                    if func == 'SEM_GROUP_BY':
                        min_commas = 2  # column, n_groups, backend
                    elif func in ['SEM_ORDER_BY', 'SEM_AGG']:
                        # These can have 2 or 3 args depending on form
                        min_commas = 1
                    
                    if comma_count < min_commas:
                        return f"{func} has insufficient arguments (expected at least {min_commas + 1})"
        
        # 6. Backend-specific template validation
        if backend == 'lotus':
            # Check for {column} syntax in semantic functions
            if any(func in query_upper for func in semantic_funcs):
                # Extract strings within semantic function calls
                pattern = r"SEM_\w+\s*\(['\"]([^'\"]*)['\"]"
                matches = re.findall(pattern, query, re.IGNORECASE)
                
                # At least one should have {column} template syntax
                has_template = any('{' in m and '}' in m for m in matches if m)
                if matches and not has_template:
                    logger.warning("LOTUS backend query may be missing {column} template syntax")
        
        elif backend == 'docetl':
            # Check for {{ input.column }} or {{ left.column }} syntax
            if any(func in query_upper for func in semantic_funcs):
                pattern = r"SEM_\w+\s*\(['\"]([^'\"]*)['\"]"
                matches = re.findall(pattern, query, re.IGNORECASE | re.DOTALL)
                
                has_jinja = any('{{' in m and '}}' in m for m in matches if m)
                if matches and not has_jinja:
                    logger.warning("DocETL backend query may be missing {{ input.column }} Jinja2 syntax")
        
        elif backend == 'palimpzest':
            # Palimpzest uses plain text, so less strict
            # Just verify SEM_ORDER_BY has column argument if present
            if 'SEM_ORDER_BY' in query_upper:
                # Match SEM_ORDER_BY with at least 2 arguments
                pattern = r'SEM_ORDER_BY\s*\(\s*([^,\)]+)\s*,'
                matches = re.findall(pattern, query, re.IGNORECASE)
                if not matches:
                    return "Palimpzest SEM_ORDER_BY must specify column as first argument"
                
                # Check that first argument looks like a column name (not a string literal)
                first_arg = matches[0].strip()
                if first_arg.startswith("'") or first_arg.startswith('"'):
                    return "Palimpzest SEM_ORDER_BY first argument must be column name, not string literal"
        
        # 7. Check for SQL keywords that might indicate errors
        error_indicators = [
            'UNDEFINED', 'ERROR', 'INVALID', 'NULL AS NULL',
            'FIXME', 'TODO', '???'
        ]
        
        for indicator in error_indicators:
            if indicator in query_upper:
                return f"Query contains error indicator: '{indicator}'"
        
        # 9. Verify no incomplete template substitutions
        if '${' in query or '{' in query and '}' not in query:
            return "Query contains incomplete template substitutions"
        
        # 9. Basic SQL syntax patterns
        # Check that SELECT is followed by something before FROM
        # More permissive to allow complex expressions and semantic functions
        select_pattern = r'SELECT\s+.+?\s+FROM'
        if not re.search(select_pattern, query, re.IGNORECASE | re.DOTALL):
            return "Invalid SELECT clause syntax"
        
        # All checks passed
        return None
    
    def _normalize_column_quotes(self, query: str) -> str:
        """
        Normalize column name quotes for DuckDB compatibility.
        Converts backticks (`) to double quotes (") for column references.
        
        Args:
            query: SQL query that may contain backtick-quoted column names
            
        Returns:
            Query with normalized double-quoted column names
        """
        import re
        
        # Replace backticks with double quotes, being careful not to replace
        # backticks inside single-quoted strings
        result = []
        in_single_quote = False
        in_backtick = False
        i = 0
        
        while i < len(query):
            char = query[i]
            
            # Track single quote context
            if char == "'" and (i == 0 or query[i-1] != '\\'):
                in_single_quote = not in_single_quote
                result.append(char)
            
            # Handle backticks (only outside single quotes)
            elif char == '`' and not in_single_quote:
                # Convert backtick to double quote
                result.append('"')
            
            else:
                result.append(char)
            
            i += 1
        
        normalized = ''.join(result)
        
        # Log if we made changes
        if '`' in query:
            logger.info(f"Normalized backticks to double quotes in query")
            logger.debug(f"Original: {query[:100]}...")
            logger.debug(f"Normalized: {normalized[:100]}...")
        
        return normalized

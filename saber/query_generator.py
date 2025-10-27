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
       SEM_SELECT('Summarize the customer {description}.', 'lotus') AS summary
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
WHERE SEM_WHERE('{rv.comment} mentions authentic or traditional cuisine', 'lotus')
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

Extract the customer sentiment (positive/negative/neutral).', 'docetl') AS sentiment
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
WHERE SEM_WHERE('Chef Experience: {{ input.c.experience }}

Return true if the chef has traditional or classical culinary training, otherwise false.', 'docetl')
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
        ]
    }
    
    SYSTEM_PROMPTS = {
        'lotus': dedent("""
        Generate SABER SQL with LOTUS backend given a database schema and natural language question.
        
        SABER SQL is a superset of standard SQL with semantic functions. All semantic functions must specify 'lotus' backend.
        
        LOTUS Semantic Functions:
        
        1. SEM_WHERE('condition', 'lotus') - Semantic filtering
           Template: Use {column_name} to reference columns in condition
           Example: WHERE SEM_WHERE('{bio} mentions machine learning', 'lotus')
        
        2. SEM_SELECT('description', 'lotus') AS alias - Semantic extraction
           Template: Use {column_name} in description
           Example: SELECT SEM_SELECT('Extract main point from {abstract}', 'lotus') AS main_point
        
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
        - Always specify 'lotus' as the backend parameter
        - Reference columns in templates using {column_name} syntax
        
        Output only the SABER SQL query, no explanations.
        """).strip(),
        
        'docetl': dedent("""
        Generate SABER SQL with DocETL backend given a database schema and natural language question.
        
        SABER SQL is a superset of standard SQL with semantic functions. All semantic functions must specify 'docetl' backend.
        
        DocETL Semantic Functions (use Jinja2 templates):
        
        1. SEM_WHERE('condition', 'docetl') - Semantic filtering
           Template: Use {{ input.column_name }} to access row values
           Example: WHERE SEM_WHERE('Bio: {{ input.bio }}\\n\\nReturn true if mentions ML, else false.', 'docetl')
        
        2. SEM_SELECT('prompt', 'docetl') AS alias - Semantic extraction
           Template: Access row data with {{ input.column_name }}
           Example: SELECT SEM_SELECT('{{ input.abstract }}\\n\\nExtract the main contribution.', 'docetl') AS contribution
        
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
        - Always specify 'docetl' as the backend parameter
        - For WHERE/SELECT, access row data via {{ input.column_name }}
        - For JOIN, use {{ left.column }} and {{ right.column }}
        
        Output only the SABER SQL query, no explanations.
        """).strip()
    }
    
    def __init__(self, model: str, api_key: Optional[str] = None, api_base: Optional[str] = None, backend: str = 'lotus'):
        """
        Initialize query generator.
        
        Args:
            model: LiteLLM model identifier (e.g., 'openai/gpt-4o-mini', 'hosted_vllm/openai/gpt-oss-20b')
            api_key: API key (optional for local models)
            api_base: API base URL (for local models)
            backend: Target backend for generated queries ('lotus' or 'docetl')
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.backend = backend
        
        if backend not in ['lotus', 'docetl']:
            # raise ValueError(f"Backend must be 'lotus' or 'docetl', got '{backend}'")
            if backend == 'palimpzest':
                logger.warning("Palimpzest backend is not supported for query generation. Defaulting to 'lotus'.")
                self.backend = 'lotus'
    
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
        if backend not in ['lotus', 'docetl']:
            raise ValueError(f"Backend must be 'lotus' or 'docetl', got '{backend}'")
        
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
        
        # Call LLM
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base
        )
        
        query = response.choices[0].message.content.strip()
        
        # Clean up common formatting issues
        if query.startswith("```sql"):
            query = query[6:]
        if query.startswith("```"):
            query = query[3:]
        if query.endswith("```"):
            query = query[:-3]
        query = query.strip()
        
        return query

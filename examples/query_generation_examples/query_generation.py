import pandas as pd
from saber import SaberEngine

# Sample data
employees_data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice Chen', 'Bob Smith', 'Carol Lee', 'David Kim', 'Eve Martinez'],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Engineering'],
    'salary': [95000, 75000, 88000, 72000, 91000],
    'bio': [
        'Alice is a senior engineer with expertise in distributed systems and cloud architecture.',
        'Bob has been with the company for 5 years selling enterprise solutions to Fortune 500 clients.',
        'Carol specializes in machine learning, data pipelines, and AI model deployment.',
        'David leads marketing campaigns focusing on digital media and brand awareness.',
        'Eve works on deep learning research, particularly in computer vision and NLP applications.'
    ]
}

products_data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['Laptop Pro', 'Office Chair', 'Wireless Mouse', 'Monitor 4K', 'Desk Lamp'],
    'category': ['Electronics', 'Furniture', 'Electronics', 'Electronics', 'Furniture'],
    'price': [1299.99, 299.99, 49.99, 599.99, 79.99],
    'description': [
        'High-performance laptop with 16GB RAM and RTX 4060. Perfect for developers and gamers.',
        'Ergonomic office chair with lumbar support and adjustable height. Very comfortable for long work sessions.',
        'Bluetooth wireless mouse with precision tracking. Works on any surface.',
        '4K UHD monitor with HDR support. Excellent color accuracy for photo and video editing.',
        'LED desk lamp with adjustable brightness. Modern design with touch controls.'
    ]
}

def main():
    # Initialize SABER engine
    engine = SaberEngine(backend='lotus', use_local_llm=False)
    # Register tables
    employees_df = pd.DataFrame(employees_data)
    products_df = pd.DataFrame(products_data)
    engine.register_table('employees', employees_df)
    engine.register_table('products', products_df)

    # Example 1: Simple semantic filtering with LOTUS
    print("Example 1: Semantic filtering (LOTUS backend)")
    print("Question: Which engineering employees have machine learning experience?\n")
    
    query1 = engine.generate(
        question="Which engineering employees have machine learning experience?",
        tables=['employees'],
        backend='lotus',
        max_sample_rows=3
    )
    print(f"Generated Query:\n{query1}\n")
    
    result1 = engine.query_ast(query1)
    print(f"Result:\n{result1}\n")
    print("-" * 80 + "\n")
    
    # Example 2: Semantic selection with ordering
    print("Example 2: Semantic projection and ordering (LOTUS backend)")
    print("Question: What are the most expensive products with positive descriptions?\n")
    
    query2 = engine.generate(
        question="What are the most expensive products with positive descriptions, ordered by price?",
        tables=['products'],
        backend='lotus',
        max_sample_rows=3
    )
    print(f"Generated Query:\n{query2}\n")
    
    result2 = engine.query_ast(query2)
    print(f"Result:\n{result2}\n")
    print("-" * 80 + "\n")
    
    # Example 3: Semantic aggregation with DocETL
    print("Example 3: Semantic aggregation (DocETL backend)")
    print("Question: Summarize employee focus areas by department\n")
    
    # Initialize DocETL backend engine
    engine_docetl = SaberEngine(backend='docetl', use_local_llm=False)
    engine_docetl.register_table('employees', employees_df)
    
    query3 = engine_docetl.generate(
        question="Summarize the main focus areas of employees in each department",
        tables=['employees'],
        backend='docetl',
        max_sample_rows=5
    )
    print(f"Generated Query:\n{query3}\n")
    
    result3 = engine_docetl.query_ast(query3)
    print(f"Result:\n{result3}\n")
    print("-" * 80 + "\n")
    
    # Example 4: Custom schema string
    print("Example 4: Using custom schema string")
    custom_schema = """
Table: orders
Columns: id (INTEGER), customer_name (TEXT), product (TEXT), amount (DECIMAL), review (TEXT)
Sample rows (3 of 10 total):
  1, 'John Doe', 'Laptop Pro', 1299.99, 'Amazing laptop, very fast and reliable'
  2, 'Jane Smith', 'Office Chair', 299.99, 'Comfortable but took a while to assemble'
  3, 'Mike Johnson', 'Monitor 4K', 599.99, 'Crystal clear display, perfect for work'
"""
    
    query4 = engine.generate(
        question="Which customers left positive reviews and spent over $500?",
        schema=custom_schema,
        backend='lotus'
    )
    print(f"Generated Query:\n{query4}\n")
    print("-" * 80 + "\n")
    
    # Example 5: Specify sample rows
    print("Example 5: Controlling sample rows")
    query5 = engine.generate(
        question="Find employees in Engineering with expertise in distributed systems",
        tables=['employees'],
        backend='lotus',
        max_sample_rows=2  # Only show 2 sample rows to LLM
    )
    print(f"Generated Query (with 2 sample rows):\n{query5}\n")
    
    result5 = engine.query_ast(query5)
    print(f"Result:\n{result5}\n")
    print("-" * 80 + "\n")
    
    # Example 6: Palimpzest backend
    print("Example 6: Palimpzest backend")
    print("Question: Which employees work on deep learning or neural networks?\n")
    
    engine_palimpzest = SaberEngine(backend='palimpzest', use_local_llm=False)
    engine_palimpzest.register_table('employees', employees_df)
    
    query6 = engine_palimpzest.generate(
        question="Which employees work on deep learning or neural networks?",
        tables=['employees'],
        backend='palimpzest',
        max_sample_rows=3
    )
    print(f"Generated Query:\n{query6}\n")
    result6 = engine_palimpzest.query_ast(query6)
    print(f"Result:\n{result6}\n")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()

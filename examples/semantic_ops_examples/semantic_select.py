import pandas as pd

from saber import SaberEngine

def demo():
    """Simple demonstration"""
    engine = SaberEngine()
    
    products = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7],
        'name': ['Golden Delicious', 'Banana', 'Red Delicious', 'Strawberry', 'Cherry', 'Blueberry', 'Cranberry'],
        'price': [12.5, 8.0, 11.0, 9.5, 15.0, 10.0, 14.0],
        'category': ['fresh', 'fresh', 'fresh', 'fresh', 'fresh', 'frozen', 'frozen'],
        'note': [
            'These are from local farms, and they are for pies.',
            'Still green, so need to wait a bit.',
            'some of them are too bitter, but some are sweet.',
            'frozen, need to keep them in the freezer.',
            'tomorrow they will be ripe.',
            'for packaging, anti-oxidants are used, so they are not fresh.',
            'they are not fresh, but they are good for smoothies.'
        ]
    })
    
    engine.register_table("products", products)

    print("Original Products DataFrame:")
    print(products.head(10))
    print()

    # Basic semantic SELECT with LOTUS
    lotus_result = engine.query_ast("""
    SELECT SEM_SELECT('What is the most famous trademark for {name}? Write only trademark.', 'lotus') AS trademark, price
    FROM products 
    WHERE SEM_WHERE('{name} is related to berry', 'lotus')
    ORDER BY price DESC
    """)
    print("SEM_SELECT for trademark of products related to 'berry':")
    print(lotus_result.head(10))
    print()

    # Basic semantic SELECT with DocETL
    docetl_result = engine.query_ast("""
    SELECT SEM_SELECT('{{ input.name }} 

Write the most famous trademark.', 'docetl') AS trademark, price
    FROM products 
    WHERE SEM_WHERE('name: {{ input.name }}
                             
Return true if name is related to berry, otherwise return false.', 'docetl')
    """)
    print("SEM_SELECT for trademark of products related to 'berry':")
    print(docetl_result.head(10))
    print()

    # Basic semantic SELECT with Palimpzest
    palimpzest_result = engine.query_ast("""
    SELECT SEM_SELECT('The most famous trademark of the name.', 'palimpzest') AS trademark, price
    FROM products 
    WHERE SEM_WHERE('name is related to berry', 'palimpzest')
    ORDER BY price DESC
    """)
    print("SEM_SELECT for trademark of products related to 'berry':")
    print(palimpzest_result.head(10))
    print()

if __name__ == "__main__":
    demo()
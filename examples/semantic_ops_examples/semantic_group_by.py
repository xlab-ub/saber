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

    # Basic semantic GROUP BY with LOTUS
    lotus_result = engine.query_ast("""
    SELECT SEM_GROUP_BY(name, 3, 'lotus') AS fruit_group, AVG(price) AS avg_price, COUNT(*) AS count
    FROM products
    GROUP BY SEM_GROUP_BY(name, 3, 'lotus')
    ORDER BY avg_price DESC
    """)
    print("SEM_GROUP_BY for products based on name:")
    print(lotus_result.head(10))
    print()

    # Basic semantic GROUP BY with DocETL
    docetl_result = engine.query_ast("""
    SELECT SEM_GROUP_BY(name, 3, 'docetl') AS fruit_group, AVG(price) AS avg_price, COUNT(*) AS count
    FROM products
    GROUP BY SEM_GROUP_BY(name, 3, 'docetl')
    ORDER BY avg_price DESC
    """)
    print("SEM_GROUP_BY for products based on name:")
    print(docetl_result.head(10))
    print()

    # # Basic semantic GROUP BY with Palimpzest
    # # Palimpzest doesn't support semantic group by.
    # # TODO: Temporarily disabled due to Palimpzest bug
    # # Issue: https://github.com/mitdbg/palimpzest/issues/185
    # palimpzest_result = engine.query_ast("""
    # SELECT SEM_GROUP_BY(name, 3, 'palimpzest') AS fruit_group, AVG(price) AS avg_price, COUNT(*) AS count
    # FROM products
    # GROUP BY SEM_GROUP_BY(name, 3, 'palimpzest')
    # ORDER BY avg_price DESC
    # """)
    # print("SEM_GROUP_BY for products based on name:")
    # print(palimpzest_result.head(10))
    # print()

if __name__ == "__main__":
    demo()
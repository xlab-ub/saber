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

    # Basic semantic AGGREGATION with LOTUS
    lotus_result = engine.query_ast("""
    SELECT SEM_AGG('Analyze the status of {note} and summarize it in one sentence.', 'lotus') AS summary
    FROM products
    """)
    print("SEM_AGGREGATION for products based on note:")
    print(lotus_result.head(10))
    print()

    # Basic semantic AGGREGATION with DocETL (Column-specific)
    docetl_result = engine.query_ast("""
    SELECT SEM_AGG(note, 'Summarize all in one sentence.', 'docetl') AS summary
    FROM products
    """)
    print("SEM_AGGREGATION for products based on note:")
    print(docetl_result.head(10))
    print()

    # Basic semantic AGGREGATION with DocETL (Global aggregation)
    docetl_result2 = engine.query_ast("""
    SELECT SEM_AGG('What are the common characteristics?', 'docetl') AS summary
    FROM products
    """)
    print("SEM_AGGREGATION for products:")
    print(docetl_result2.head(10))
    print()

    # # Basic semantic AGGREGATION with Palimpzest (Column-specific)
    # # TODO: Temporarily disabled due to validation issues with Palimpzest aggregation
    # palimpzest_result = engine.query_ast("""
    # SELECT SEM_AGG(note, 'Summarize all in one sentence.', 'palimpzest') AS summary
    # FROM products
    # """)
    # print("SEM_AGGREGATION for products based on note:")
    # print(palimpzest_result.head(10))
    # print()

    # # Basic semantic AGGREGATION with Palimpzest (Global aggregation)
    # # TODO: Temporarily disabled due to validation issues with Palimpzest aggregation
    # palimpzest_result2 = engine.query_ast("""
    # SELECT SEM_AGG('What are the common characteristics?', 'palimpzest') AS summary
    # FROM products
    # """)
    # print("SEM_AGGREGATION for products:")
    # print(palimpzest_result2.head(10))
    # print()

    # Basic semantic AGGREGATION with GROUP BY with LOTUS
    lotus_result2 = engine.query_ast("""
    SELECT SEM_AGG('Write single representative word for each {name}.', 'lotus') AS summary
    FROM products
    GROUP BY SEM_GROUP_BY(name, 3, 'lotus')
    """)
    print("SEM_AGGREGATION with SEM_GROUP_BY for products based on name:")
    print(lotus_result2.head(10))
    print()

    # Basic semantic AGGREGATION with GROUP BY with DocETL (Column-specific)
    docetl_result3 = engine.query_ast("""
    SELECT SEM_AGG(name, 'Write single representative word for each name.', 'docetl') AS summary
    FROM products
    GROUP BY SEM_GROUP_BY(name, 3, 'docetl')
    """)
    print("SEM_AGGREGATION with SEM_GROUP_BY for products based on name:")
    print(docetl_result3.head(10))
    print()

    # Basic semantic AGGREGATION with GROUP BY with DocETL (Global aggregation)
    docetl_result4 = engine.query_ast("""
    SELECT SEM_AGG('Describe the characteristics of each group.', 'docetl') AS summary
    FROM products
    GROUP BY SEM_GROUP_BY(name, 3, 'docetl')
    """)
    print("SEM_AGGREGATION with SEM_GROUP_BY for products based on name:")
    print(docetl_result4.head(10))
    print()


if __name__ == "__main__":
    demo()
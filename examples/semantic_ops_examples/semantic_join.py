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

    desserts = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Berry Pie', 'Banana Split', 'Cherry Cake'],
        'price': [15.0, 10.0, 12.0],
        'note': [
            'Delicious pie with fresh berries.',
            'Classic dessert with ripe bananas.',
            'Cake with fresh strawberries and cream.'
        ]
    })

    sections = pd.DataFrame({
        'id': [1, 2],
        'name': ['fresh', 'frozen'],
        'note': ['Fresh fruits and vegetables', 'Frozen fruits and vegetables']
    })
    
    engine.register_table("products", products)
    engine.register_table("desserts", desserts)
    engine.register_table("sections", sections)

    print("Original Products DataFrame:")
    print(products.head(10))
    print()
    print("Original Desserts DataFrame:")
    print(desserts.head(10))
    print()
    print("Original Sections DataFrame:")
    print(sections.head(10))
    print()

    # Basic semantic JOIN with LOTUS
    lotus_result = engine.query_ast("""
    SELECT products.name AS product_name, desserts.name AS dessert_name
    FROM SEM_JOIN(products, desserts, '{name:left} is generally used in {name:right}', 'lotus')
    """)
    print("SEM_JOIN for products and desserts:")
    print(lotus_result.head(10))
    print()

    # Basic semantic JOIN with DocETL
    docetl_result = engine.query_ast("""
    SELECT products.name AS product_name, desserts.name AS dessert_name
    FROM SEM_JOIN(products, desserts, 'Compare the following product and dessert:
Product Name: {{ left.name }}
                               
Dessert Name: {{ right.name }}

Is this product a good match for the dessert? Respond with True if it is a good match or False if it is not a suitable match.', 'docetl')
    """)
    print("SEM_JOIN for products and desserts:")
    print(docetl_result.head(10))
    print()

    # Basic semantic JOIN with Palimpzest
    palimpzest_result = engine.query_ast("""
    SELECT products.name AS product_name, desserts.name AS dessert_name
    FROM SEM_JOIN(products, desserts, 'The product is a good match for the dessert.', 'palimpzest')
    """)
    print("SEM_JOIN for products and desserts:")
    print(palimpzest_result.head(10))
    print()

    # Basic semantic JOIN with another table JOIN with LOTUS
    lotus_result2 = engine.query_ast("""
    SELECT products.name AS product_name, desserts.name AS dessert_name, sections.note AS section_note
    FROM SEM_JOIN(products, desserts, '{name:left} is generally used in {name:right}', 'lotus') AS products_desserts
    JOIN sections ON sections.name = products_desserts.category
    """)
    print("SEM_JOIN for products and desserts with JOIN sections:")
    print(lotus_result2.head(10))
    print()

    # Basic semantic JOIN with another table JOIN with DocETL
    docetl_result2 = engine.query_ast("""
    SELECT products.name AS product_name, desserts.name AS dessert_name, sections.note AS section_note
    FROM SEM_JOIN(products, desserts, 'Compare the following product and dessert:
Product Name: {{ left.name }}
                               
Dessert Name: {{ right.name }}

Is this product a good match for the dessert? Respond with True if it is a good match or False if it is not a suitable match.', 'docetl') AS products_desserts
    JOIN sections ON sections.name = products_desserts.category
    """)
    print("SEM_JOIN for products and desserts with JOIN sections:")
    print(docetl_result2.head(10))
    print()

    # Basic semantic JOIN with another table JOIN with Palimpzest
    palimpzest_result2 = engine.query_ast("""
    SELECT products.name AS product_name, desserts.name AS dessert_name, sections.note AS section_note
    FROM SEM_JOIN(products, desserts, 'The product is a good match for the dessert.', 'palimpzest') AS products_desserts
    JOIN sections ON sections.name = products_desserts.category
    """)
    print("SEM_JOIN for products and desserts with JOIN sections:")
    print(palimpzest_result2.head(10))
    print()


if __name__ == "__main__":
    demo()
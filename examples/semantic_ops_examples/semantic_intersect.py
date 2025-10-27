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
            'Delicious pie with fresh apples.',
            'Classic dessert with ripe bananas.',
            'Cake with fresh strawberries and cream.'
        ]
    })
    
    engine.register_table("products", products)
    engine.register_table("desserts", desserts)

    print("Original Products DataFrame:")
    print(products.head(10))
    print()
    print("Original Desserts DataFrame:")
    print(desserts.head(10))
    print()

    # Basic semantic INTERSECT
    result = engine.query_ast("""
    SELECT *
    FROM SEM_INTERSECT(
        "SELECT name FROM products",
        "SELECT name FROM desserts"
    )
    """)
    print("SEM_INTERSECT for products and desserts:")
    print(result.head(10))
    print()


if __name__ == "__main__":
    demo()
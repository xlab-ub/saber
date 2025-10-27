import pandas as pd

from saber import SaberEngine

def unified_query_demo():
    # Unified Query Abstraction Demo - Restaurants and Chefs
    print("--- Unified Query Abstraction Demo (Restaurants & Chefs) ---")
    print()
    print("Natural Language Question:")
    print("Which restaurants focus on authentic or traditional cuisine and are led by chefs who have traditional training or family background in cooking? Show me the top 2 highest-rated restaurants with their chef names, cuisine types, ratings, and what cooking specialties or unique techniques each chef is known for.")

    restaurants = pd.DataFrame({
        'name': ['Sakura Garden', 'Mediterranean Breeze', 'Spice Route', 'Farm to Table', 'Fusion Palace', 'Ocean Pearl'],
        'location': ['Tokyo District', 'Coastal Boulevard', 'Downtown', 'Countryside', 'City Center', 'Harbor View'],
        'cuisine_type': ['Japanese', 'Mediterranean', 'Indian', 'Farm-to-table American', 'Asian Fusion', 'Seafood'],
        'rating': [4.8, 4.6, 4.7, 4.5, 4.9, 4.4],
        'description': [
            'An authentic Japanese restaurant specializing in traditional sushi and ramen, featuring fresh ingredients flown in daily from Tokyo fish markets.',
            'A Mediterranean restaurant offering healthy dishes with olive oil, fresh herbs, and grilled seafood, emphasizing heart-healthy cooking methods.',
            'An Indian restaurant known for complex spice blends and aromatic curries, using traditional tandoor cooking and regional recipes passed down through generations.',
            'A restaurant committed to using locally sourced organic ingredients, supporting sustainable farming practices and offering seasonal menus.',
            'An innovative Asian fusion restaurant combining techniques from Chinese, Thai, and Korean cuisines with modern presentation and creative flavor combinations.',
            'A seafood restaurant specializing in fresh catch prepared with simple techniques to highlight natural flavors, featuring daily specials based on market availability.'
        ],
        'chef_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006']
    })

    chefs = pd.DataFrame({
        'chef_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
        'name': ['Hiroshi Tanaka', 'Maria Konstantinidis', 'Rajesh Sharma', 'Emma Johnson', 'David Kim', 'Carlos Martinez'],
        'experience': [
            'Trained at the prestigious Tsuji Culinary Institute in Osaka, worked under Michelin-starred chefs in Tokyo for 12 years, specializes in traditional Edomae sushi techniques.',
            'Studied culinary arts in Athens, Greece, and worked in family restaurants across the Mediterranean islands. Expert in using olive oil and fresh herbs to create healthy, flavorful dishes.',
            'Born in Mumbai, learned traditional cooking from his grandmother, later studied at culinary school in Delhi. Master of spice blending and tandoor cooking with 15 years of experience.',
            'Graduate of Culinary Institute of America, passionate about sustainable farming and organic ingredients. Worked on organic farms before opening her restaurant.',
            'Korean-American chef who studied both in Seoul and Paris, combining traditional Asian techniques with French culinary methods to create innovative fusion dishes.',
            'Third-generation fisherman turned chef, learned seafood preparation from his father and grandfather. Expert in selecting the freshest catch and simple preparation methods.'
        ]
    })
    
    backend_free_query = """
    SELECT r.name AS restaurant, c.name AS chef, r.cuisine_type, r.rating,
           SEM_SELECT('Summarize the chef specialty or unique cooking technique mentioned in their experience.') AS chef_specialty
    FROM restaurants AS r JOIN chefs AS c ON r.chef_id = c.chef_id
    WHERE SEM_WHERE('the chef has traditional training or family background in cooking.') AND 
          SEM_WHERE('the restaurant focuses on authentic or traditional cuisine.')
    ORDER BY CAST(r.rating AS FLOAT) DESC
    LIMIT 2;
    """

    # --- LOTUS Backend ---
    print("\n--- Testing with LOTUS backend ---")
    engine_lotus = SaberEngine(backend='lotus')
    engine_lotus.register_table("restaurants", restaurants)
    engine_lotus.register_table("chefs", chefs)
    result_lotus, stats_lotus = engine_lotus.query_ast_with_benchmark(backend_free_query, reset=True)
    print("Result from LOTUS backend:")
    print(result_lotus)
    print(f"\nBenchmark Stats for LOTUS:")
    print(f"  Total Cost:          ${stats_lotus.total_cost:.6f}")
    print(f"  Total Time:          {stats_lotus.total_execution_time_seconds:.2f}s")
    print(f"  Semantic Time:       {stats_lotus.total_semantic_execution_time_seconds:.2f}s")
    print(f"  SQL Time:            {stats_lotus.total_non_semantic_execution_time_seconds:.2f}s")
    print(f"  Rewriting Time:      {stats_lotus.total_rewriting_time_seconds:.2f}s")

    # --- DocETL Backend ---
    print("\n--- Testing with DocETL backend ---")
    engine_docetl = SaberEngine(backend='docetl')
    engine_docetl.register_table("restaurants", restaurants)
    engine_docetl.register_table("chefs", chefs)
    result_docetl, stats_docetl = engine_docetl.query_ast_with_benchmark(backend_free_query, reset=True)
    print("\nResult from DocETL backend:")
    print(result_docetl)
    print(f"\nBenchmark Stats for DocETL:")
    print(f"  Total Cost:          ${stats_docetl.total_cost:.6f}")
    print(f"  Total Time:          {stats_docetl.total_execution_time_seconds:.2f}s")
    print(f"  Semantic Time:       {stats_docetl.total_semantic_execution_time_seconds:.2f}s")
    print(f"  SQL Time:            {stats_docetl.total_non_semantic_execution_time_seconds:.2f}s")
    print(f"  Rewriting Time:      {stats_docetl.total_rewriting_time_seconds:.2f}s")

    # --- Palimpzest Backend ---
    print("\n--- Testing with Palimpzest backend ---")
    engine_palimpzest = SaberEngine(backend='palimpzest')
    engine_palimpzest.register_table("restaurants", restaurants)
    engine_palimpzest.register_table("chefs", chefs)
    result_palimpzest, stats_palimpzest = engine_palimpzest.query_ast_with_benchmark(backend_free_query, reset=True)
    print("\nResult from Palimpzest backend:")
    print(result_palimpzest)
    print(f"\nBenchmark Stats for Palimpzest:")
    print(f"  Total Cost:          ${stats_palimpzest.total_cost:.6f}")
    print(f"  Total Time:          {stats_palimpzest.total_execution_time_seconds:.2f}s")
    print(f"  Semantic Time:       {stats_palimpzest.total_semantic_execution_time_seconds:.2f}s")
    print(f"  SQL Time:            {stats_palimpzest.total_non_semantic_execution_time_seconds:.2f}s")
    print(f"  Rewriting Time:      {stats_palimpzest.total_rewriting_time_seconds:.2f}s")

if __name__ == "__main__":
    unified_query_demo()

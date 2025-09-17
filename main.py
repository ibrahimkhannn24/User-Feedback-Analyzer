# main.py

import os
from dotenv import load_dotenv
from scraper import scrape_and_save_reviews
from graph_builder import build_knowledge_graph
from query_engine import answer_query_with_graph

load_dotenv() 

def main():
    print("--- Welcome to the Voice of the Customer AI ---")
    
    app_name = input("Enter the name of the app you want to analyze (e.g., 'Spotify'): ")
    country = input("Enter the 2-letter country code for the App Store (e.g., 'us'): ")
    app_id = input(f"Enter the Apple App ID for '{app_name}' (e.g., 324684580 for Spotify): ")
    sanitized_app_name = app_name.lower().replace(" ", "_")
    
    reviews_dir = os.path.join("reviews", sanitized_app_name)
    index_path = os.path.join(reviews_dir, "graphrag_index")

    # Check if a cached index exists
    if os.path.exists(index_path):
        print(f"Loading existing GraphRAG index for '{app_name}' from '{index_path}'...")
    else:
        print("No index found. Starting the full pipeline...")
        # Step 1: Scrape reviews
        scraped_dir = scrape_and_save_reviews(app_name, app_id, country, review_count=200)
        
        if not scraped_dir:
            print("Could not proceed without reviews. Exiting.")
            return

        # Step 2: Build all GraphRAG artifacts
        success = build_knowledge_graph(scraped_dir, index_path)
        if not success:
            print("Failed to build the knowledge graph index. Exiting.")
            return

    # Step 3: Query Loop
    print("\nGraphRAG index is ready. You can now ask questions.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\n> Ask a question about the reviews: ")
        if query.lower() == 'exit':
            break
        
        print("AI is thinking...")
        answer = answer_query_with_graph(query, index_path, reviews_dir)
        print("\n--- AI Insight ---")
        print(answer)
        print("--------------------")

if __name__ == "__main__":
    main()
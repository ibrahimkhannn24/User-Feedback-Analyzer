# User Feedback Analyzer
User Feedback Analyzer is a powerful tool for extracting insights from user reviews using Graph-based Retrieval-Augmented Generation (GraphRAG). It combines web scraping, knowledge graph building, and AI-driven query answering to help product teams understand user sentiment, feature requests, and bug reports.

## Key Features
1. App Store Review Scraper

    - Uses Selenium to scrape user reviews from the Apple App Store.
    - Handles dynamic content loading and modal-based review extraction.
    - Saves reviews to text files for further processing.

2. Knowledge Graph Builder

    - Extracts entities (e.g., feature requests, bug reports, sentiment) and relationships from reviews using LLMs (Generative AI).
    - Builds a graph representation of the extracted data using NetworkX.
    - Detects communities within the graph and generates concise summaries of each community.
    
3. Vector Search with FAISS

    - Embeds entities as vectors using Google’s text-embedding models.
    - Efficiently retrieves relevant nodes and relationships using FAISS for semantic search.
    
4. GraphRAG Query Engine

    - Combines graph data, community summaries, and original reviews to provide detailed answers to user queries.
    - Leverages Generative AI to synthesize professional and actionable responses.

## How It Works
1. Scrape Reviews
    - Input the app name, country code, and App Store ID, and scrape user reviews using Selenium.

2. Build Knowledge Graph
    - Reviews are processed to extract entities and relationships, which are stored in a knowledge graph. Communities are identified and summarized.

3. Query the Graph
    - Ask natural language questions about the reviews (e.g., "What bugs are users reporting?"). The query engine retrieves relevant information from the graph and generates a comprehensive answer.

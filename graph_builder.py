# graph_builder.py

import os
import json
import time
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import faiss
import pickle

load_dotenv()

# --- LLM Configuration (Extraction & Summarization) ---
generation_config = {"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 4096}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    llm_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except KeyError:
    print("Error: GOOGLE_API_KEY not found. Please ensure it is set in your .env file.")
    exit()

def extract_entities_from_review(review_text):
    # This function's code is unchanged.
    prompt = f"""
    Analyze the following user review and extract key entities and their relationships.
    The entities to extract are:
    - FEATURE_REQUEST: A specific feature the user is asking for.
    - BUG_REPORT: An issue or bug the user is reporting.
    - USER_SENTIMENT: The overall sentiment of the review (e.g., "Positive", "Negative", "Mixed", "Neutral").
    - PRODUCT_COMPONENT: A specific part of the app mentioned (e.g., "UI", "Playlist", "Search", "Login").

    Return the output as a JSON object with two keys: "entities" and "relationships".
    Example:
    Review: "The new update is terrible. The app crashes every time I open my playlist. I wish there was a dark mode."
    Output:
    {{
      "entities": [
        {{"id": "app_crash", "type": "BUG_REPORT", "value": "App crashes on opening playlist"}},
        {{"id": "playlist_feature", "type": "PRODUCT_COMPONENT", "value": "Playlist"}},
        {{"id": "dark_mode_request", "type": "FEATURE_REQUEST", "value": "Dark mode"}},
        {{"id": "negative_sentiment", "type": "USER_SENTIMENT", "value": "Negative"}}
      ],
      "relationships": [
        {{"source": "app_crash", "target": "playlist_feature", "type": "related_to"}},
        {{"source": "negative_sentiment", "target": "app_crash", "type": "describes"}},
        {{"source": "negative_sentiment", "target": "dark_mode_request", "type": "describes"}}
      ]
    }}

    Now, analyze this review:
    ---
    {review_text}
    ---
    """
    try:
        response = llm_model.generate_content(prompt)
        response_text = response.text
        import re
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_text = match.group(0)
            return json.loads(json_text)
        else:
            print("Could not find a JSON block in the LLM response.")
            return None
    except Exception as e:
        print(f"Could not parse LLM response as JSON or other API error: {e}")
        return None

def detect_and_store_communities(G):
    print("Detecting communities in the knowledge graph...")
    undirected_G = G.to_undirected()
    communities = list(greedy_modularity_communities(undirected_G))
    community_map = {node: i for i, community_nodes in enumerate(communities) for node in community_nodes}
    nx.set_node_attributes(G, community_map, 'community_id')
    print(f"Found {len(communities)} communities and tagged all nodes.")
    return G, communities

def generate_community_summaries(G, communities):
    print("Generating summaries for each community...")
    summaries = {}
    for i, community_nodes in enumerate(tqdm(communities, desc="Summarizing Communities")):
        community_data = []
        for node_id in community_nodes:
            node_data = G.nodes[node_id]
            community_data.append(f"- Entity: {node_data.get('value')} (Type: {node_data.get('type')})")
        
        context_str = "\n".join(community_data)
        prompt = f"""
        The following is a list of entities and concepts belonging to a single community detected within a knowledge graph of app reviews.
        Summarize the main theme or topic of this community in a single, concise sentence.

        Entities:
        {context_str}

        Summary:
        """
        try:
            response = llm_model.generate_content(prompt)
            summaries[i] = response.text.strip()
            time.sleep(0.5) # Rate limiting
        except Exception as e:
            print(f"Could not generate summary for community {i}: {e}")
            summaries[i] = "Summary generation failed."
            
    return summaries

def create_entity_embeddings_index(G):
    print("Creating vector embeddings for all graph entities...")
    node_ids = []
    node_values = []
    for node_id, data in G.nodes(data=True):
        if 'value' in data:
            node_ids.append(node_id)
            node_values.append(data['value'])

    if not node_values:
        print("No node values to embed.")
        return None, None

    try:
        # Use the text-embedding model for batch embedding
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=node_values,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result['embedding']
        embeddings = np.array(embeddings).astype('float32')
        
        # Create and train FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        print(f"Successfully created FAISS index with {len(node_ids)} entity embeddings.")
        return index, node_ids
    except Exception as e:
        print(f"Could not create embeddings or FAISS index: {e}")
        return None, None

def build_knowledge_graph(reviews_dir, index_path):
    """
    Builds all GraphRAG artifacts: graph, communities, summaries, and vector index.
    """
    G = nx.MultiDiGraph()
    review_files = [f for f in os.listdir(reviews_dir) if f.endswith(".txt")]

    # --- Step 1: Build Initial Graph ---
    for filename in tqdm(review_files, desc="Building Knowledge Graph"):
        filepath = os.path.join(reviews_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            review_text = f.read()
        extracted_data = extract_entities_from_review(review_text)
        if extracted_data:
            for entity in extracted_data.get("entities", []):
                G.add_node(entity["id"], type=entity["type"], value=entity["value"], source_file=filename)
            for rel in extracted_data.get("relationships", []):
                if G.has_node(rel["source"]) and G.has_node(rel["target"]):
                    G.add_edge(rel["source"], rel["target"], type=rel["type"])
        time.sleep(1)

    print(f"Initial graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    if G.number_of_nodes() == 0:
        return False

    # --- Step 2: Detect Communities ---
    G, communities = detect_and_store_communities(G)

    # --- Step 3: Generate Community Summaries ---
    community_summaries = generate_community_summaries(G, communities)

    # --- Step 4: Create Entity Embeddings ---
    faiss_index, faiss_node_ids = create_entity_embeddings_index(G)
    if faiss_index is None:
        return False

    # --- Step 5: Save all artifacts ---
    os.makedirs(index_path, exist_ok=True)
    with open(os.path.join(index_path, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    with open(os.path.join(index_path, "community_summaries.json"), "w") as f:
        json.dump(community_summaries, f)
    faiss.write_index(faiss_index, os.path.join(index_path, "entity_embeddings.faiss"))
    with open(os.path.join(index_path, "faiss_node_ids.json"), "w") as f:
        json.dump(faiss_node_ids, f)
        
    print(f"All GraphRAG artifacts saved to '{index_path}'.")
    return True
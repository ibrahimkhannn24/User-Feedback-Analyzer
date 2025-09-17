# query_engine.py

import os
import json
import pickle
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import faiss

load_dotenv()

# --- LLM for Answering Questions (Generation) ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    generation_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
except KeyError:
    print("Error: GOOGLE_API_KEY not found. Please ensure it is set in your .env file.")
    exit()

def retrieve_and_build_context(query, G, faiss_index, faiss_node_ids, summaries, reviews_dir, top_k=5):
    """
    Implements the Local Search workflow to build a rich context.
    """
    # --- Step 1: Similar Entity Search ---
    print("Step 1: Performing semantic search for entry point entities...")
    try:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        query_embedding = np.array([query_embedding]).astype('float32')
    except Exception as e:
        print(f"Could not embed query: {e}")
        return ""

    distances, indices = faiss_index.search(query_embedding, top_k)
    entry_point_node_ids = [faiss_node_ids[i] for i in indices[0]]
    print(f"Found entry points: {entry_point_node_ids}")

    # --- Step 2: Context Augmentation ---
    context = {"local_graph": [], "global_community": [], "source_text": []}
    seen_communities = set()
    seen_source_files = set()

    for node_id in entry_point_node_ids:
        node_data = G.nodes[node_id]
        
        # A. Local Graph Context (Fanning out to neighbors)
        local_info = f"Entity '{node_data.get('value')}' (Type: {node_data.get('type')}) is related to:"
        neighbors_info = []
        for neighbor_id in G.neighbors(node_id):
            neighbor_data = G.nodes[neighbor_id]
            neighbors_info.append(f"  - '{neighbor_data.get('value')}' (Type: {neighbor_data.get('type')})")
        if not neighbors_info:
            neighbors_info.append("  - No direct relationships found.")
        context["local_graph"].append(f"{local_info}\n" + "\n".join(neighbors_info))

        # B. Global Community Context
        community_id = node_data.get('community_id')
        if community_id is not None and community_id not in seen_communities:
            summary = summaries.get(str(community_id), "No summary available.")
            context["global_community"].append(f"This topic belongs to a community summarized as: '{summary}'")
            seen_communities.add(community_id)

        # C. Source Text Context
        source_file = node_data.get('source_file')
        if source_file and source_file not in seen_source_files:
            try:
                filepath = os.path.join(reviews_dir, source_file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    review_text = f.read()
                context["source_text"].append(f"--- START OF RELEVANT REVIEW ({source_file}) ---\n{review_text}\n--- END OF REVIEW ---")
                seen_source_files.add(source_file)
            except FileNotFoundError:
                pass # Ignore if file not found

    # --- Step 3: Assemble Final Context String ---
    final_context = "CONTEXT FOR YOUR ANSWER:\n\n"
    if context["global_community"]:
        final_context += "## Overall Topic Summaries\n" + "\n".join(context["global_community"]) + "\n\n"
    if context["local_graph"]:
        final_context += "## Specific Entity Relationships\n" + "\n\n".join(context["local_graph"]) + "\n\n"
    if context["source_text"]:
        final_context += "## Grounding Source Text from Original Reviews\n" + "\n\n".join(context["source_text"])
        
    return final_context

def answer_query_with_graph(query, index_path, reviews_dir):
    """
    Answers a user's query using the full GraphRAG Local Search pipeline.
    """
    # Load all the pre-built artifacts
    try:
        with open(os.path.join(index_path, "graph.pkl"), "rb") as f:
            G = pickle.load(f)
        with open(os.path.join(index_path, "community_summaries.json"), "r") as f:
            summaries = json.load(f)
        faiss_index = faiss.read_index(os.path.join(index_path, "entity_embeddings.faiss"))
        with open(os.path.join(index_path, "faiss_node_ids.json"), "r") as f:
            faiss_node_ids = json.load(f)
    except FileNotFoundError:
        return "Could not find the necessary index files. Please build the index first."

    # --- RETRIEVAL STEP ---
    context_text = retrieve_and_build_context(query, G, faiss_index, faiss_node_ids, summaries, reviews_dir)
    if not context_text:
        return "I couldn't find any relevant information in the knowledge graph to answer your question."
    
    # --- GENERATION STEP ---
    prompt = f"""
    You are an AI assistant for a product manager. Your task is to answer questions based on a knowledge graph built from user reviews.
    Use the provided context, which includes global summaries, local entity relationships, and original review text, to synthesize a comprehensive and actionable answer.
    Do not mention the internal mechanics (e.g., "based on the community summary"). Answer the question directly and professionally.

    {context_text}
    ---

    USER'S QUESTION:
    "{query}"

    ANSWER:
    """
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"
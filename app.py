import gradio as gr
import numpy as np 
import pandas as pd
import random
import math

# Mock Graph Embedding Data Simulation 
# Using simple 3D vectors to simulate closeness in vector space means high chemical similarity.
# Vector dimensions (simulated): [Sweetness, Aromatic Intensity, Acidity/Savoriness]
INGREDIENT_VECTORS = {
    "Strawberry": np.array([0.95, 0.70, 0.40]), 
    "Basil": np.array([0.80, 0.90, 0.15]),      
    "Goat Cheese": np.array([0.60, 0.85, 0.10]),
    "Black Pepper": np.array([0.05, 0.50, 0.05]), 
    "Chocolate": np.array([0.40, 0.25, 0.90]),
    "Blue Cheese": np.array([0.00, 0.95, 0.45]),
    "Tomato": np.array([0.65, 0.35, 0.85]),     
    "Vanilla": np.array([0.99, 0.10, 0.00]),  
    "Coconut": np.array([0.75, 0.55, 0.15]),
    "Garlic": np.array([0.10, 0.05, 0.00]),
}

# Core Logic (Graph Embedding Similarity) 
def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors (range -1.0 to 1.0)."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_pairings(ingredient):
    """
    Simulates finding flavor pairings using vector similarity from Graph Embeddings.
    Returns: list of (pairing_name, score) tuples.
    """
    normalized_input = ingredient.strip().title()
    
    if normalized_input not in INGREDIENT_VECTORS:
        return None 
    
    input_vector = INGREDIENT_VECTORS[normalized_input]
    similarities = {}
    
    for name, vector in INGREDIENT_VECTORS.items():
        if name != normalized_input: 
            score = cosine_similarity(input_vector, vector)
            similarities[name] = score

    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    top_matches_with_scores = sorted_similarities[:5]
    
    return top_matches_with_scores

def generate_output(user_input):
    """
    The main Gradio prediction function.
    Returns: results_markdown (str), recipe_markdown (str)
    """
    pairings_with_scores = get_pairings(user_input)
    
    if not pairings_with_scores:
        return (f"Could not find a vector (embedding) for **'{user_input.title()}'** in our current mock database. Try a known ingredient!", 
                "")

    # Flavor Pairings Output 
    results_markdown = f"## Flavor Pairings for: {user_input.title()}"
    results_markdown += "\n\n| Rank | Pairing Ingredient | Similarity Score |"
    results_markdown += "\n| :---: | :--- | :---: |"
    
    for i, (name, score) in enumerate(pairings_with_scores):
        results_markdown += f"\n| **{i+1}** | **{name.title()}** | `{score:.4f}` |"
        
    results_markdown += "\n\n*The score reflects the numerical closeness of the vector embeddings.*"

    # Recipe Suggestion Output 
    pair1, score1 = pairings_with_scores[0]
    pair2, score2 = pairings_with_scores[min(1, len(pairings_with_scores) - 1)] 
    
    recipe_markdown = f"""
    ## Culinary Genius Suggestion (Top Score: {score1:.3f})
    
    **Dish Concept:** A unique {user_input.title()} tart with a {pair1.title()} base and a {pair2.title()} dust.
    
    **Why it Works (The Science):** The **Graph Embeddings Model** placed these ingredients closely together in the vector space, indicating a high overlap in their aromatic profiles.
    """
    
    return results_markdown, recipe_markdown

# Gradio Interface 

# Input component (Text box for ingredient)
ingredient_input = gr.Textbox(
    label="Enter an ingredient (e.g., Strawberry, Basil, Chocolate):", 
    value="Strawberry"
)

# Output components (Results)
pairings_output = gr.Markdown(label="Flavor Pairings Results")
recipe_output = gr.Markdown(label="Practical Recipe Concept")

gr.Interface(
    # The function to run when the user clicks 'Submit'
    fn=generate_output,

    inputs=ingredient_input,

    outputs=[pairings_output, recipe_output],
    
    # Title and description for the interface
    title="🧪 The Flavor Lab: Graph Embeddings MVP",
    description="Welcome to The Flavor Lab, running on a **simulated Graph Embeddings Model** Input any ingredient below. The model converts ingredient relationships (aromatic compounds) into vectors and finds the closest matches.",
    
    theme=gr.themes.Soft(),
    allow_flagging="never",
).launch(
    server_name="0.0.0.0", 
    server_port=7860
)

import streamlit as st
import numpy as np 
import pandas as pd
import random
import math 

# Mock Graph Embedding Data Simulation 
# Using simple 3D vectors to simulate the concept where closeness in vector space means high chemical similarity.
# Vector dimensions (simulated): [Sweetness, Aromatic Intensity, Acidity/Savoriness]
INGREDIENT_VECTORS = {
    "Strawberry": np.array([0.95, 0.70, 0.40]), 
    "Basil": np.array([0.80, 0.90, 0.15]),      
    "Goat Cheese": np.array([0.60, 0.85, 0.10]),# High similarity with Basil, moderate with Strawberry
    "Black Pepper": np.array([0.05, 0.50, 0.05]), 
    "Chocolate": np.array([0.40, 0.25, 0.90]),  # Bitter, Fatty, Complex
    "Blue Cheese": np.array([0.00, 0.95, 0.45]), # High similarity with Chocolate (oddly)
    "Tomato": np.array([0.65, 0.35, 0.85]),     
    "Vanilla": np.array([0.99, 0.10, 0.00]),  
    "Coconut": np.array([0.75, 0.55, 0.15]), # High aromatic, low acidity
    "Garlic": np.array([0.10, 0.05, 0.00]),
}

# Graph Embedding Similarity

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors (range -1.0 to 1.0)."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    # Handle division by zero if vectors are zero-length (shouldn't happen)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_pairings(ingredient):
    """
    Simulates finding flavor pairings using vector similarity from Graph Embeddings.
    Returns: list of (pairing_name, score) tuples.
    """
    normalized_input = ingredient.strip().title()
    
    # Check if the input ingredient has a known vector (embedding)
    if normalized_input not in INGREDIENT_VECTORS:
        # Simple normalization check for common variations
        if normalized_input == 'Strawberry':
             input_vector = INGREDIENT_VECTORS['Strawberry'] 
        else:
            return None 
    else:
        input_vector = INGREDIENT_VECTORS[normalized_input]
    
    similarities = {}
    
    # Calculate Similarity to all other ingredients (nodes)
    for name, vector in INGREDIENT_VECTORS.items():
        if name != normalized_input: # Don't pair an ingredient with itself
            # Calculate the chemical similarity score (0.0 to 1.0 in our simulated data)
            score = cosine_similarity(input_vector, vector)
            similarities[name] = score

    # Rank and select the top matches
    # Sort the dictionary items by score in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    # Select the top 5 most similar ingredients and their scores
    top_matches_with_scores = sorted_similarities[:5]
    
    return top_matches_with_scores

def get_recipe_suggestion(primary_ingredient, pairings_with_scores):
    """Generates a simple recipe suggestion based on the top pairing."""
    if not pairings_with_scores:
        return f"Try pairing {primary_ingredient} with something unusual, like Pomegranate or Cardamom!"

    # Use the top pairing (highest similarity score)
    pair1, score1 = pairings_with_scores[0]
    
    # Use the second pairing for contrast/complexity if available
    pair2, score2 = pairings_with_scores[min(1, len(pairings_with_scores) - 1)] 
    
    return f"""
    ### Culinary Genius Suggestion (Similarity Score: {score1:.3f})
    
    **Dish Concept:** A unique {primary_ingredient.title()} tart with a {pair1.title()} base and a {pair2.title()} dust.
    
    **Why it Works (The Science):** The **Graph Embeddings Model** placed these ingredients closely together in the vector space (Similarity Score: {score1:.3f}), indicating a high overlap in their aromatic profiles. This ensures a harmonious blend that hits the **Scientific** and **Amazing** criteria of The Flavor Lab.
    """


# Streamlit UI 

st.set_page_config(
    page_title="MVP of Graph Embeddings for Flavors",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🧪 The Flavor Lab: Graph Embeddings MVP")
st.markdown("""
Welcome to The Flavor Lab, running on a **simulated Graph Embeddings Model**! 
Input any ingredient below to unlock scientifically-perfect flavor pairings. 
*The model converts ingredient relationships (aromatic compounds) into vectors 
and finds the closest matches in that vector space.*
""")

# Input section
st.subheader("1. Input Ingredient")
user_input = st.text_input(
    "Enter an ingredient (e.g., Strawberry, Basil, Chocolate):", 
    value="Strawberry",
    key="ingredient_input"
)

if user_input:
    st.markdown("---")
    st.subheader(f"2. Analyzing: '{user_input.title()}'")
    
    # Run the core logic
    pairings_with_scores = get_pairings(user_input)
    
    if pairings_with_scores:
        st.success("Analysis Complete! Graph Embeddings Model ranked pairings.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Flavor Pairings (Ranked by Chemical Similarity)")
            st.markdown(f"**Primary Ingredient:** **{user_input.title()}**")
            
            # Display pairings and scores
            st.markdown("#### Top Matches (Highest Similarity Score is best):")
            for name, score in pairings_with_scores:
                st.info(f"**{name.title()}** (Score: {score:.4f})")
        
        with col2:
            st.markdown("### 3. Practical Recipe Suggestion")
            st.markdown(get_recipe_suggestion(user_input, pairings_with_scores))
            st.markdown("*This suggestion integrates the highest-ranked pairing from the vector similarity search.*")

    else:
        st.warning(f"Could not find a vector (embedding) for '{user_input.title()}' in our current mock database. Try a known ingredient!")

st.markdown("---")
st.markdown("""
*MVP* of *Similarity Search Model** built upon the principles of **Graph Embeddings**
""")

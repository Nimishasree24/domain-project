# Dual-Mode Explainable AI System for ESG Greenwashing Detection: Company-Level and Custom Text Analysis with SHAP Transparency

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import pickle
import nltk
import os

# --- FIX FOR STREAMLIT CLOUD NLTK DATA ---
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# List of required NLTK resources and their expected names/locations
# app.py (Modified NLTK Data Setup)

# ... (Lines 1-27: NLTK path setup)

# List of required NLTK resources and their expected names/locations
nltk_packages = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger", 
    "omw-1.4": "corpora/omw-1.4"
}

for pkg, resource_path in nltk_packages.items():
    try:
        # Try to find the resource by its path first
        nltk.data.find(resource_path)
    except LookupError:
        # If not found, download the package
        
        # --- CHANGE MADE HERE ---
        # Use print() instead of st.info() and st.success() 
        print(f"Downloading required NLTK package: {pkg}...") 
        try:
            nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
            print(f"{pkg} downloaded successfully.")
        except Exception as e:
            # Use a print statement or st.error if you want errors displayed, 
            # but using print keeps everything in the logs.
            print(f"Failed to download {pkg}. Error: {e}")

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Streamlit Config 
st.set_page_config(page_title="ESG Greenwashing Detector (Explainable)", layout="wide")
st.title("Dual-Mode Explainable AI System for ESG Greenwashing Detection: Company-Level and Custom Text Analysis with SHAP Transparency")
st.markdown("Analyze and interpret ESG statements for potential greenwashing using AI explainability (SHAP).")

# Paths 
MODEL_PATH = r"model.pkl"
VECT_PATH = r"vectorizer.pkl"
ESG_CSV_PATH = r"esg_documents_for_dax_companies.csv"

# Preprocessing 
lemmatizer = WordNetLemmatizer()

def get_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN
# app.py (around line 56)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # --- FIX APPLIED HERE ---
    # Original (Line 56): words = word_tokenize(text)
    words = word_tokenize(text, preserve_line=True) 
    # --- END FIX ---
    
    pos_tags = nltk.pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(w, get_pos(t)) for w, t in pos_tags]
    return " ".join(lemmatized)


# Load Files 
@st.cache_data
def load_model_vectorizer(model_path, vect_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vect_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load model/vectorizer: {e}")
        return None, None

@st.cache_data
def load_esg_csv(path):
    try:
        df = pd.read_csv(path, sep="|", engine="python", quoting=3, on_bad_lines="skip")
        df["content"] = df["content"].astype(str)
        return df
    except Exception as e:
        st.warning(f"Failed to load ESG CSV: {e}")
        return pd.DataFrame()

# SHAP Compatibility Helper 
def make_explainer(model, background):
    """Handle SHAP version compatibility automatically."""
    try:
        return shap.LinearExplainer(model, background, feature_perturbation="interventional")
    except TypeError:
        return shap.LinearExplainer(model, background, feature_dependence="independent")

# Load Everything 
model, vectorizer = load_model_vectorizer(MODEL_PATH, VECT_PATH)
esg_docs = load_esg_csv(ESG_CSV_PATH)
companies = sorted(esg_docs["company"].dropna().unique()) if not esg_docs.empty else []

# Mode Selection 
mode = st.radio("Select Mode:", ["Analyze Existing Company", "Custom ESG Statement"], horizontal=True)

# Function for Prediction + SHAP + Interpretation 
def analyze_statement(text):
    clean_stmt = clean_text(text)
    vec = vectorizer.transform([clean_stmt])

    # Input validation
    if vec.nnz == 0:
        st.warning("No recognizable ESG-related content found — please enter a meaningful statement.")
        return

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    # Display prediction 
    if prob >= 0.6:
        st.error(f"Potential Greenwashing — Probability: {prob:.2%}")
    else:
        st.success(f"Low Greenwashing Signal — Probability: {prob:.2%}")

    # SHAP Explainability 
    st.markdown("## Explainable AI Insights (SHAP)")

    # Use small random sample from ESG data as SHAP background
    if not esg_docs.empty:
        background_texts = esg_docs["content"].dropna().sample(min(50, len(esg_docs)), random_state=42)
        background = vectorizer.transform(background_texts)
    else:
        background = vec  # fallback

    explainer = make_explainer(model, background)
    shap_values = explainer(vec)

    # Extract word importances
    feature_names = vectorizer.get_feature_names_out()
    importance = shap_values.values[0]
    top_indices = np.argsort(np.abs(importance))[-10:]

    # Handle empty / zero SHAP output safely
    if np.all(importance == 0):
        st.warning("SHAP could not find meaningful feature influences (possibly too short input).")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E57373" if v > 0 else "#81C784" for v in importance[top_indices]]
    sns.barplot(x=importance[top_indices], y=feature_names[top_indices], palette=colors, ax=ax)
    ax.set_title("Top words influencing prediction (SHAP)")
    ax.set_xlabel("SHAP value (impact on model output)")
    st.pyplot(fig)

    # Interpretation 
    st.markdown("### Model Interpretation (Auto-Generated)")
    top_positive = [feature_names[i] for i in top_indices if importance[i] > 0]
    top_negative = [feature_names[i] for i in top_indices if importance[i] < 0]
    pos_words = ", ".join(top_positive[:3]) if top_positive else "None"
    neg_words = ", ".join(top_negative[:3]) if top_negative else "None"

    if prob >= 0.6:
        verdict = "**High Greenwashing likelihood**"
        explanation = (
            f"Promotional or vague ESG cues such as *{pos_words}* increased the greenwashing probability "
            f"({prob*100:.2f}%). Authentic indicators (e.g., *{neg_words}*) had less influence."
        )
    elif 0.4 <= prob < 0.6:
        verdict = "**Moderate Greenwashing signal**"
        explanation = (
            f"The text shows a mix of genuine and possibly misleading ESG language. "
            f"Words like *{pos_words}* contributed toward greenwashing, while *{neg_words}* balanced it."
        )
    else:
        verdict = "**Low Greenwashing signal**"
        explanation = (
            f"Although some words such as *{pos_words}* slightly suggested greenwashing, "
            f"factual or policy-related terms like *{neg_words}* reduced the overall risk "
            f"to {prob*100:.2f}%. The text appears authentic."
        )

    st.markdown(verdict)
    st.info(explanation)

# Analyze Existing Company
if mode == "Analyze Existing Company":
    st.header("Analyze ESG Statement from Existing Company")

    if not companies:
        st.error("No companies found in ESG dataset.")
    else:
        selected = st.selectbox("Select Company:", companies)
        if selected:
            row = esg_docs[esg_docs["company"].str.contains(selected, case=False, na=False)].iloc[0]
            content = row.get("content", "")
            st.write("### ESG Statement Preview:")
            st.write(content[:1000] + ("..." if len(content) > 1000 else ""))

            if st.button("Predict"):
                analyze_statement(content)

# Custom ESG Statement 
else:
    st.header("Enter Custom ESG Statement")
    cname = st.text_input("Enter Company Name")
    statement = st.text_area("Enter ESG Statement")

    if st.button("Predict"):
        if not cname.strip() or not statement.strip():
            st.error("Please provide both company name and ESG statement.")
        else:
            analyze_statement(statement)

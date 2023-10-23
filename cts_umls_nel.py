# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:09:56 2023

@author: vmurc
"""

import streamlit as st
import pandas as pd
from umls_query_functions import create_dataframe,plot_histogram,make_radio_graph,plot_semantic_types_plotly
from config import UMLS_API_KEY

pd.set_option('display.width', 100)

# Set the page layout to wide format
st.set_page_config(layout="wide")

# Define your UMLS API key, desired parameters, and function to query UMLS
api_key = UMLS_API_KEY
sabs = "MSH,SNOMEDCT_US,RXNORM,ICD10,CMT,NCIT,LNC,ICD9CM,ICF,NDFRT,MEDCIN,HL7V3.1"  # Comma-separated list of vocabularies
ttys = "PT,SY,IN"  # Comma-separated list of term types

st.title("UMLS Query App")

# Input field for the user to enter a term
input_term = st.text_input("Enter a term to query the UMLS")

# Initialize session_state
if 'use_ui' not in st.session_state:
    st.session_state.use_ui = True
    
# Wrap the entire function with st.form
with st.form("umls_query_form"):
    # Use a toggle for selecting version and store its state
    use_ui = st.toggle("View CUI Graph",  st.session_state.use_ui)

    # Button to trigger the query
    if st.form_submit_button("Query UMLS"):
        # Check if the input term is provided
        if not input_term:
            st.warning("Please enter a term.")
        else:
            # Call the function to query UMLS and create the DataFrame
            df = create_dataframe(api_key, input_term, sabs, ttys)
            
            st.write("## Results")
            st.dataframe(df) 
            
            # Create two columns for layout
            col1, col2, col3 = st.columns([0.8,1,0.8])
             
            # Create a Matplotlib figure and plot the histogram in the right column
            with col1:
                st.write("## Histogram")
                fig = plot_histogram(df)
                st.plotly_chart(fig)

            with col2:
                # Display the directed graph
                st.write("## Directed Graph Visualization")
                fig = make_radio_graph(input_term, df, use_ui)
                st.plotly_chart(fig,use_container_width = True)
            
            with col3:
                st.write("## Semantic Types")
                fig = plot_semantic_types_plotly(df)
                st.plotly_chart(fig)
# Arrange descriptions in three columns and two rows using markdown
st.write("## String Comparison Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### Cosine Similarity")
    st.write("Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space.")
    st.write(r'$\text{Cosine Similarity}(X, Y) = \frac{{X \cdot Y}}{{\|X\| \|Y\|}}$')
    st.write("Variables:")
    st.write("- $X, Y$: TF-IDF vectors of the input strings.")
    st.write("- $\|X\|, \|Y\|$: Euclidean norms of the vectors $X$ and $Y$.")

with col2:
    st.write("### Normalized Levenshtein Distance")
    st.write("Normalized Levenshtein distance is a measure of the minimum number of single-character edits (insertions, deletions, substitutions) needed to change one string into another.")
    st.write(r'$\text{Normalized Levenshtein Distance}(s1, s2) = \frac{{\text{Levenshtein Distance}(s1, s2)}}{{\max(\text{len}(s1), \text{len}(s2))}}$')
    st.write("Variables:")
    st.write("- $s1, s2$: Input strings.")
    st.write("- $\text{Levenshtein Distance}(s1, s2)$: Minimum edit distance between $s1$ and $s2$.")

with col3:
    st.write("### Jaccard Similarity")
    st.write("Jaccard similarity measures the size of the intersection of two sets divided by the size of the union of the sets.")
    st.write(r'$\text{Jaccard Similarity}(A, B) = \frac{{|A \cap B|}}{{|A \cup B|}}$')
    st.write("Variables:")
    st.write("- $A, B$: Sets of unique words in input strings.")

col4, col5, col6 = st.columns(3)

with col4:
    st.write("### Normalized Hamming Distance")
    st.write("Normalized Hamming distance counts the number of positions at which the corresponding characters are different and normalizes it by the length of the input string.")
    st.write(r'$\text{Normalized Hamming Distance}(s1, s2) = \frac{{\text{Hamming Distance}(s1, s2)}}{{\text{len}(s1)}}$')
    st.write("Variables:")
    st.write("- $s1, s2$: Input strings.")
    st.write("- $\text{Hamming Distance}(s1, s2)$: Number of differing characters between $s1$ and $s2$.")

with col5:
    st.write("### Jaro-Winkler Similarity")
    st.write("Jaro-Winkler similarity measures the similarity between two strings by comparing the common characters and transpositions.")
    st.write(r'$\text{Jaro-Winkler Similarity}(s1, s2) = \frac{{\text{Jaro-Winkler Distance}(s1, s2)}}{{\max(\text{len}(s1), \text{len}(s2))}}$')
    st.write("Variables:")
    st.write("- $s1, s2$: Input strings.")
    st.write("- $\text{Jaro-Winkler Distance}(s1, s2)$: Measure of similarity between $s1$ and $s2$.")

with col6:
    st.write("### Dice Coefficient")
    st.write("Dice coefficient (Bigram similarity) measures the similarity between two strings based on the intersection and union of their bigrams.")
    st.write(r'$\text{Dice Coefficient}(s1, s2) = \frac{{2 \cdot |s1 \cap s2|}}{{|s1| + |s2|}}$')
    st.write("Variables:")
    st.write("- $s1, s2$: Input strings.")
    st.write("- $|s1 \cap s2|$: Number of common bigrams.")
    st.write("- $|s1|, |s2|$: Number of bigrams in $s1$ and $s2$.")

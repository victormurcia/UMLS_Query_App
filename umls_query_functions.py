# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:08:49 2023

@author: vmurc
"""
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
from nltk.util import ngrams
import jellyfish
from similarity.jarowinkler import JaroWinkler
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def query_umls_api(api_key, input_string, sabs, ttys, version="current"):
    base_uri = 'https://uts-ws.nlm.nih.gov'

    # Initialize an empty list to store the results
    results_list = []
    
    def get_semantic_types_for_cui(cui):
        path = '/content/' + version + '/CUI/' + cui
        query = {'apiKey': api_key}
        output = requests.get(base_uri + path, params=query)
        output.encoding = 'utf-8'
        
        outputJson = output.json()
        if 'semanticTypes' in outputJson['result']:
            return [x['name'] for x in outputJson['result']['semanticTypes']]
        else:
            return []
        
    page = 0

    while True:
        page += 1
        path = '/search/' + version
        query = {'string': input_string, 'apiKey': api_key, 'rootSource': sabs, 'termType': ttys, 'pageNumber': page}
        output = requests.get(base_uri + path, params=query)
        output.encoding = 'utf-8'

        outputJson = output.json()
        results = (([outputJson['result']])[0])['results']

        if len(results) == 0:
            if page == 1:
                print('No results found for ' + input_string + '\n')
            break

        for item in results:
            semantic_types = get_semantic_types_for_cui(item['ui'])
            results_list.append({
                'UI': item['ui'],
                'Name': item['name'],
                'URI': item['uri'],
                'Source': item['rootSource'],
                'SemanticTypes': ', '.join(semantic_types)
            })

    return results_list

# Cosine Similarity (TF-IDF-based)
def cosine_similarity_tf_idf(str1, str2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([str1, str2])
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_similarities[0][0]

# Levenshtein (Edit) Distance
def levenshtein_distance(str1, str2):
    return Levenshtein.distance(str1, str2)

# Jaccard Similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Hamming Distance
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must have equal length for Hamming distance")
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

# Jaro-Winkler Similarity
def jaro_winkler_similarity(str1, str2):
    return jellyfish.jaro_winkler(str1, str2)

# Dice Coefficient (Bigram Similarity)
def dice_coefficient(str1, str2):
    bigrams1 = set(ngrams(str1, 2))
    bigrams2 = set(ngrams(str2, 2))
    intersection = len(bigrams1.intersection(bigrams2))
    union = len(bigrams1) + len(bigrams2)
    return (2.0 * intersection) / union if union > 0 else 0

# Soundex Encoding
def soundex_encoding(str1, str2):
    return jellyfish.soundex(str1), jellyfish.soundex(str2)

# Metaphone Encoding
def metaphone_encoding(str1, str2):
    return jellyfish.metaphone(str1), jellyfish.metaphone(str2)

def calculate_string_metrics(input_string, results_list):
    metrics = {
        'Levenshtein': [],
        'Jaccard': [],
        'Cosine': [],
        'Hamming': [],
        'Jaro_Winkler': [],  # Renamed metric
        'Sorensen-Dice': [],
    }

    jarowinkler = JaroWinkler()  # Initialize JaroWinkler

    # Convert the input string to lowercase
    input_string = input_string.lower()

    for result in results_list:
        name = result['Name'].lower()

        # Calculate various string comparison metrics
        levenshtein_dist = Levenshtein.distance(input_string, name)
        max_length = max(len(input_string), len(name))
        
        # Calculate the normalized Levenshtein distance
        normalized_levenshtein_dist = 1 - levenshtein_dist / max_length
        metrics['Levenshtein'].append(normalized_levenshtein_dist)

        input_set = set(input_string.split())
        name_set = set(name.split())
        metrics['Jaccard'].append(jaccard_similarity(input_set, name_set))
        metrics['Cosine'].append(cosine_similarity_tf_idf(input_string, name))

        # Calculate the normalized Hamming distance
        # Pad the shorter string with spaces to ensure equal length
        max_length = max(len(input_string), len(name))
        input_string_padded = input_string.ljust(max_length)
        name_padded = name.ljust(max_length)

        # Calculate the Hamming distance
        hamming_dist = hamming_distance(input_string_padded, name_padded)
        normalized_hamming_dist = 1 - hamming_dist / len(input_string)
        metrics['Hamming'].append(normalized_hamming_dist)

        # Calculate the Jaro-Winkler similarity using the jarowinkler library
        metrics['Jaro_Winkler'].append(jarowinkler.similarity(input_string, name))  # Removed normalization

        metrics['Sorensen-Dice'].append(dice_coefficient(input_string, name))

    return metrics

@st.cache_data
def make_radio_graph(input_term, df, use_ui=False):
    # Create a radar plot
    if use_ui:
        name_col = 'UI'
    else:
        name_col = 'Name'
    fig = px.line_polar(df, r='Sum_of_Sims',
                        #text='Sum_of_Sims', 
                        theta=name_col,
                        template="plotly_dark",
                        line_close=True,
                        hover_name="Name")
    
    # Customize the radar plot appearance (optional)
    fig.update_traces(fill='toself',
                      hoverinfo='theta+text+r',
                      line=dict(color='rebeccapurple'))
    fig.update_traces(hoverlabel=dict(namelength=-1)) 
    
    fig.update_layout(
        width=1200,  # Set the width of the plot
        height=600,  # Set the height of the plot
        #polar=dict(radialaxis=dict(visible=True, tickfont=dict(size=12))),  # Show radial axes
        showlegend=False,  # Hide legend
        title="Radar Plot of Similarity Scores",
        )
    
    return fig

def plot_semantic_types_plotly(df):
    # Count the occurrences of each SemanticType
    semantic_type_counts = df['SemanticTypes'].value_counts().reset_index()
    semantic_type_counts.columns = ['SemanticType', 'Count']
    
    # Get the 'rocket' color palette from Seaborn
    num_types = len(semantic_type_counts['SemanticType'])
    colors = sns.color_palette('rocket', num_types).as_hex()
    
    # Create the bar chart using plotly express
    fig = px.bar(semantic_type_counts, 
                 y='SemanticType', 
                 x='Count', 
                 width=600,
                 height=600,
                 orientation='h', 
                 title='Count of Semantic Types',
                 labels={'Count': 'Count', 'SemanticType': 'Semantic Type'},
                 color='SemanticType', 
                 color_discrete_sequence=colors)
    # Hide the color legend since it's redundant
    fig.update_layout(showlegend=False)
    
    return fig

def plot_histogram(df):
    fig = px.histogram(df, x='Sum_of_Sims', nbins=6)
    
    # Customizations
    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        width=500,
        height=600,
        font=dict(size=14,),
    )
    fig.update_traces(marker_color='darkcyan', marker_line=dict(width=1, color='black'))
    return fig

@st.cache_data
def create_dataframe(api_key, input_string, sabs, ttys):
    # Query the UMLS API
    results_list = query_umls_api(api_key, input_string, sabs, ttys)
    # Calculate various string comparison metrics
    metrics = calculate_string_metrics(input_string, results_list)

    # Create a Pandas DataFrame from the results list
    df = pd.DataFrame(results_list)

    # Add columns for each string comparison metric
    for metric_name, metric_values in metrics.items():
        df[metric_name] = metric_values
    
    # Add a column for the sum of all similarity metrics
    df['Sum_of_Sims'] = df[['Levenshtein', 
                                    'Jaccard', 
                                    'Cosine',
                                    'Hamming', 
                                    'Jaro_Winkler', 
                                    'Sorensen-Dice']].sum(axis=1)

    return df

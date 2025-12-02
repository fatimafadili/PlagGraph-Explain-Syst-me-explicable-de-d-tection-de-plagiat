# app.py - Application Streamlit Complète PlagGraph-Explain
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import json
from datetime import datetime
import base64
from io import BytesIO

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="PlagGraph-Explain",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    /* En-tête principal */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sous-titres */
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* Cartes de métriques */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #1E3A8A;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Badges de décision */
    .plagiarism-high {
        color: #DC2626;
        font-weight: 800;
        background: linear-gradient(135deg, #FEE2E2 0%, #FCA5A5 100%);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        border: 2px solid #DC2626;
        box-shadow: 0 4px 6px rgba(220, 38, 38, 0.2);
    }
    
    .plagiarism-moderate {
        color: #D97706;
        font-weight: 800;
        background: linear-gradient(135deg, #FEF3C7 0%, #FBBF24 100%);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        border: 2px solid #D97706;
        box-shadow: 0 4px 6px rgba(217, 119, 6, 0.2);
    }
    
    .plagiarism-low {
        color: #059669;
        font-weight: 800;
        background: linear-gradient(135deg, #D1FAE5 0%, #34D399 100%);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        border: 2px solid #059669;
        box-shadow: 0 4px 6px rgba(5, 150, 105, 0.2);
    }
    
    /* Boutons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 15px 25px;
        font-weight: 600;
        color: #475569;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e2e8f0;
        color: #1E3A8A;
        border-color: #cbd5e1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white !important;
        border-color: #1D4ED8;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 2px solid #e2e8f0;
    }
    
    /* Widgets */
    .stSlider > div > div {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Sélecteurs */
    .stSelectbox, .stTextArea, .stTextInput {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox:hover, .stTextArea:hover, .stTextInput:hover {
        border-color: #3B82F6;
    }
    
    /* Cards d'info */
    .info-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #0ea5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(14, 165, 233, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSES DU SYSTÈME
# ============================================================================
class AdvancedTextPreprocessor:
    def __init__(self):
        self.stop_words_fr = set([
            'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 
            'que', 'qui', 'dans', 'en', 'pour', 'avec', 'sur', 'par', 'au', 
            'aux', 'ce', 'cette', 'ces', 'son', 'sa', 'ses', 'mon', 'ton', 
            'notre', 'votre', 'leur', 'leurs', 'ma', 'ta', 'mes', 'tes', 
            'nos', 'vos', 'eux', 'elle', 'elles', 'lui', 'ils', 'je', 'tu', 
            'il', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'y', 
            'à', 'a', 'ou', 'où', 'donc', 'or', 'ni', 'car', 'mais', 'si'
        ])
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words_fr and len(word) > 2]
        return ' '.join(words)
    
    def split_into_sentences(self, text):
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

class MultiMethodSimilarityCalculator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def calculate_tfidf_cosine(self, text1, text2):
        vectorizer = TfidfVectorizer()
        try:
            vectors = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def calculate_jaccard_similarity(self, text1, text2):
        set1 = set(text1.split())
        set2 = set(text2.split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def calculate_ngram_similarity(self, text1, text2, n=3):
        tokens1 = text1.split()
        tokens2 = text2.split()
        if len(tokens1) < n or len(tokens2) < n:
            return 0.0
        
        ngrams1 = set([' '.join(tokens1[i:i+n]) for i in range(len(tokens1)-n+1)])
        ngrams2 = set([' '.join(tokens2[i:i+n]) for i in range(len(tokens2)-n+1)])
        
        if not ngrams1 or not ngrams2:
            return 0.0
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        return intersection / union if union > 0 else 0.0
    
    def calculate_lcs_similarity(self, text1, text2):
        words1 = text1.split()
        words2 = text2.split()
        m, n = len(words1), len(words2)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif words1[i-1] == words2[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        
        lcs_length = L[m][n]
        avg_length = (m + n) / 2
        return lcs_length / avg_length if avg_length > 0 else 0.0
    
    def calculate_edit_similarity(self, text1, text2):
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def calculate_all_similarities(self, doc1, doc2):
        proc1 = self.preprocessor.preprocess(doc1)
        proc2 = self.preprocessor.preprocess(doc2)
        
        tfidf = self.calculate_tfidf_cosine(proc1, proc2)
        jaccard = self.calculate_jaccard_similarity(proc1, proc2)
        ngram2 = self.calculate_ngram_similarity(proc1, proc2, n=2)
        ngram3 = self.calculate_ngram_similarity(proc1, proc2, n=3)
        ngram4 = self.calculate_ngram_similarity(proc1, proc2, n=4)
        lcs = self.calculate_lcs_similarity(proc1, proc2)
        edit = self.calculate_edit_similarity(proc1, proc2)
        
        weights = {'tfidf': 0.3, 'jaccard': 0.15, 'ngram2': 0.1, 
                   'ngram3': 0.15, 'ngram4': 0.1, 'lcs': 0.1, 'edit': 0.1}
        combined = (tfidf * weights['tfidf'] + jaccard * weights['jaccard'] +
                   ngram2 * weights['ngram2'] + ngram3 * weights['ngram3'] +
                   ngram4 * weights['ngram4'] + lcs * weights['lcs'] + 
                   edit * weights['edit'])
        
        # Décision de plagiat
        if combined >= 0.7:
            decision = "PLAGIAT ÉLEVÉ"
            level = "Niveau de plagiat élevé"
            confidence = "Très élevée (>95%)"
            color = "#EF4444"
        elif combined >= 0.5:
            decision = "SIMILARITÉ MODÉRÉE"
            level = "Similarité significative détectée"
            confidence = "Élevée (80-95%)"
            color = "#F59E0B"
        elif combined >= 0.3:
            decision = "SIMILARITÉ FAIBLE"
            level = "Quelques similarités détectées"
            confidence = "Modérée (60-80%)"
            color = "#10B981"
        else:
            decision = "NON PLAGIAT"
            level = "Très faible similarité"
            confidence = "Faible (<60%)"
            color = "#10B981"
        
        len1 = len(proc1.split())
        len2 = len(proc2.split())
        normalized_length = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        return {
            'tfidf_cosine': tfidf,
            'jaccard': jaccard,
            'ngram_2': ngram2,
            'ngram_3': ngram3,
            'ngram_4': ngram4,
            'lcs': lcs,
            'edit_distance': edit,
            'combined_score': combined,
            'plagiarism_decision': decision,
            'plagiarism_level': level,
            'confidence': confidence,
            'normalized_length': normalized_length,
            'color': color
        }

class PlagiarismVisualizer:
    def __init__(self):
        self.colors = {
            'plagiarism_high': '#EF4444',
            'plagiarism_moderate': '#F59E0B',
            'plagiarism_low': '#10B981',
            'methods': ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444']
        }
    
    def create_similarity_radar(self, scores_dict, doc_names):
        methods = ['TF-IDF', 'Jaccard', 'N-gram (2)', 'LCS', 'Edit Distance']
        scores = [
            scores_dict.get('tfidf_cosine', 0),
            scores_dict.get('jaccard', 0),
            scores_dict.get('ngram_2', 0),
            scores_dict.get('lcs', 0),
            scores_dict.get('edit_distance', 0)
        ]
        
        scores_closed = scores + [scores[0]]
        methods_closed = methods + [methods[0]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=methods_closed,
            fill='toself',
            name=f'{doc_names[0]} vs {doc_names[1]}',
            line=dict(color=self.colors['methods'][0], width=3),
            fillcolor='rgba(59, 130, 246, 0.3)',
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=methods,
            mode='markers+text',
            marker=dict(size=12, color='white', line=dict(width=2, color='darkblue')),
            text=[f'{s:.3f}' for s in scores],
            textposition='top center',
            textfont=dict(size=11, color='black'),
            name='Valeurs',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    ticktext=['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    tickfont=dict(size=10),
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=90
                )
            ),
            showlegend=True,
            title=dict(
                text=f'Radar des Similarités: {doc_names[0]} ↔ {doc_names[1]}',
                font=dict(size=18, color='#1E3A8A')
            ),
            height=400
        )
        
        return fig
    
    def create_specific_pairs_comparison(self, all_results):
        specific_pairs = ['doc_1_doc_2', 'doc_1_doc_4', 'doc_2_doc_4']
        filtered_results = [r for r in all_results if r['pair_id'] in specific_pairs]
        
        pairs = []
        scores = []
        colors = []
        
        for result in filtered_results:
            pairs.append(f"{result['doc1']}↔{result['doc2']}")
            score = result['combined_score']
            scores.append(score)
            
            if score >= 0.7:
                colors.append(self.colors['plagiarism_high'])
            elif score >= 0.5:
                colors.append(self.colors['plagiarism_moderate'])
            else:
                colors.append(self.colors['plagiarism_low'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pairs,
            y=scores,
            text=[f"{s:.3f}" for s in scores],
            textposition='outside',
            textfont=dict(size=14, color='black'),
            marker_color=colors,
            marker_line=dict(color='darkgray', width=1),
            hovertemplate='<b>%{x}</b><br>Score combiné: %{y:.3f}<extra></extra>',
            width=0.6
        ))
        
        fig.update_layout(
            title=dict(
                text='Comparaison des Scores Combinés - Paires Sélectionnées',
                font=dict(size=18, color='#1E3A8A')
            ),
            xaxis_title="Paires de Documents",
            yaxis_title="Score Combiné",
            yaxis=dict(
                range=[0, 1],
                tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            height=400
        )
        
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Seuil Plagiat Élevé (0.7)")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                     annotation_text="Seuil Plagiat Modéré (0.5)")
        
        return fig
    
    def create_specific_heatmap(self, all_results):
        specific_docs = ['doc_1', 'doc_2', 'doc_4']
        n = len(specific_docs)
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, 1.0)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    doc1 = specific_docs[i]
                    doc2 = specific_docs[j]
                    pair_result = next((r for r in all_results 
                                      if (r['doc1'] == doc1 and r['doc2'] == doc2) or
                                      (r['doc1'] == doc2 and r['doc2'] == doc1)), None)
                    if pair_result:
                        matrix[i][j] = pair_result['combined_score']
        
        doc_labels = []
        for doc_id in specific_docs:
            if doc_id == 'doc_1':
                doc_labels.append('Doc1\nIA Original')
            elif doc_id == 'doc_2':
                doc_labels.append('Doc2\nIA Plagié')
            else:
                doc_labels.append('Doc4\nIA Réécrit')
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=doc_labels,
            y=doc_labels,
            colorscale='RdYlGn_r',
            zmin=0,
            zmax=1,
            text=[[f"{val:.3f}" for val in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            hovertemplate='<b>%{y} ↔ %{x}</b><br>Score combiné: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Matrice de Similarité (Scores Combinés)',
                font=dict(size=18, color='#1E3A8A')
            ),
            xaxis_title="Document Cible",
            yaxis_title="Document Source",
            height=400,
            width=500
        )
        
        return fig
    
    def create_methods_comparison_with_combined(self, scores_dict):
        methods = ['TF-IDF', 'Jaccard', 'N-gram (2)', 'LCS', 'Edit Distance', 'SCORE COMBINÉ']
        values = [
            scores_dict.get('tfidf_cosine', 0),
            scores_dict.get('jaccard', 0),
            scores_dict.get('ngram_2', 0),
            scores_dict.get('lcs', 0),
            scores_dict.get('edit_distance', 0),
            scores_dict.get('combined_score', 0)
        ]
        
        colors = self.colors['methods'] + ['#7C3AED']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='outside',
            textfont=dict(size=12, color='black'),
            marker_color=colors,
            marker_line=dict(color='darkgray', width=1),
            hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>',
            width=0.7
        ))
        
        fig.update_layout(
            title=dict(
                text='Comparaison des 5 Méthodes + Score Combiné',
                font=dict(size=18, color='#1E3A8A')
            ),
            xaxis_title="Méthodes de Similarité",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_combined_score_gauge(self, combined_score, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=combined_score,
            title={'text': f"SCORE COMBINÉ", 'font': {'size': 20}},
            number={
                'font': {'size': 40, 'color': color},
                'valueformat': '.3f',
                'suffix': ''
            },
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': color, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.7], 'color': "lightyellow"},
                    {'range': [0.7, 1], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': combined_score
                }
            }
        ))
        
        fig.update_layout(height=400, margin=dict(t=80, b=50, l=50, r=50))
        return fig

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def analyze_all_pairs(documents, calculator, preprocessor):
    """Analyse toutes les paires de documents"""
    all_results = []
    
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            doc1 = documents[i]
            doc2 = documents[j]
            
            scores = calculator.calculate_all_similarities(doc1['content'], doc2['content'])
            
            similar_segments = []
            sentences1 = preprocessor.split_into_sentences(doc1['content'])
            sentences2 = preprocessor.split_into_sentences(doc2['content'])
            
            for sent1 in sentences1:
                for sent2 in sentences2:
                    if len(sent1) > 10 and len(sent2) > 10:
                        sent_scores = calculator.calculate_all_similarities(sent1, sent2)
                        if sent_scores['combined_score'] > 0.7:
                            similar_segments.append({
                                'segment1': sent1[:100] + "..." if len(sent1) > 100 else sent1,
                                'segment2': sent2[:100] + "..." if len(sent2) > 100 else sent2,
                                'similarity': sent_scores['combined_score']
                            })
            
            proc1 = preprocessor.preprocess(doc1['content'])
            proc2 = preprocessor.preprocess(doc2['content'])
            words1 = Counter(proc1.split())
            words2 = Counter(proc2.split())
            
            common_words = {}
            for word in set(words1.keys()).union(set(words2.keys())):
                freq1 = words1.get(word, 0)
                freq2 = words2.get(word, 0)
                if freq1 >= 2 and freq2 >= 2:
                    common_words[word] = {'doc1_freq': freq1, 'doc2_freq': freq2, 'total': freq1 + freq2}
            
            sorted_common = dict(sorted(common_words.items(), key=lambda x: x[1]['total'], reverse=True))
            
            result = {
                'pair_id': f"{doc1['id']}_{doc2['id']}",
                'doc1': doc1['id'],
                'doc2': doc2['id'],
                'doc1_title': doc1['title'],
                'doc2_title': doc2['title'],
                **scores,
                'similar_segments_count': len(similar_segments),
                'common_words_count': len(common_words),
                'similar_segments': similar_segments[:3],
                'top_common_words': list(sorted_common.items())[:10]
            }
            
            all_results.append(result)
    
    return all_results

def create_download_link(content, filename, file_type="text/plain"):
    """Crée un lien de téléchargement"""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:{file_type};base64,{b64}" download="{filename}">📥 {filename}</a>'

def generate_json_report(result):
    """Génère un rapport JSON"""
    report = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "system": "PlagGraph-Explain v2.0",
            "pair_id": result['pair_id']
        },
        "documents": {
            "doc1": {
                "id": result['doc1'],
                "title": result['doc1_title']
            },
            "doc2": {
                "id": result['doc2'],
                "title": result['doc2_title']
            }
        },
        "scores": {
            "tfidf_cosine": result['tfidf_cosine'],
            "jaccard": result['jaccard'],
            "ngram_2": result['ngram_2'],
            "ngram_3": result['ngram_3'],
            "ngram_4": result['ngram_4'],
            "lcs": result['lcs'],
            "edit_distance": result['edit_distance'],
            "combined_score": result['combined_score']
        },
        "decision": {
            "verdict": result['plagiarism_decision'],
            "level": result['plagiarism_level'],
            "confidence": result['confidence'],
            "color": result['color']
        },
        "statistics": {
            "similar_segments_count": result['similar_segments_count'],
            "common_words_count": result['common_words_count']
        }
    }
    return json.dumps(report, indent=2, ensure_ascii=False)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🔍 PlagGraph-Explain</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6B7280; margin-bottom: 2rem;">Système Explicable de Détection de Plagiat avec IA + Visualisations Avancées</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ **Paramètres d'Analyse**")
        
        # Sélection de l'exemple
        st.markdown("#### 📊 Exemples prédéfinis")
        example = st.selectbox(
            "Charger un exemple",
            ["Sélectionner...", "Plagiat Évident (IA)", "Plagiat Modéré (IA)", "Non Plagiat (Différents domaines)"],
            key="example_selector"
        )
        
        # Seuils
        st.markdown("#### 🎯 Seuils de décision")
        col1, col2 = st.columns(2)
        with col1:
            moderate_threshold = st.slider(
                "Seuil modéré",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Score minimum pour plagiat modéré"
            )
        with col2:
            high_threshold = st.slider(
                "Seuil élevé",
                min_value=0.1,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="Score minimum pour plagiat élevé"
            )
        
        # Méthodes
        st.markdown("#### 🔧 Méthodes activées")
        use_tfidf = st.checkbox("TF-IDF Cosine", value=True)
        use_jaccard = st.checkbox("Jaccard", value=True)
        use_ngram = st.checkbox("N-gram", value=True)
        use_lcs = st.checkbox("LCS", value=True)
        use_edit = st.checkbox("Edit Distance", value=True)
        
        st.markdown("---")
        
        # À propos
        st.markdown("### ℹ️ **À propos**")
        with st.expander("Voir les détails"):
            st.info("""
            **PlagGraph-Explain** est un système avancé qui combine :
            
            ✅ **5 méthodes de similarité** :
            - TF-IDF Cosine (sémantique)
            - Jaccard (lexicale)
            - N-gram (séquentielle)
            - LCS (sous-séquences)
            - Edit Distance (édition)
            
            ✅ **Visualisations interactives** :
            - Radar charts
            - Heatmaps
            - Graphiques comparatifs
            - Jaunes de score
            
            ✅ **Fonctionnalités avancées** :
            - Détection de segments similaires
            - Analyse de mots communs
            - Rapports exportables
            - Interface intuitive
            """)
    
    # Initialisation
    preprocessor = AdvancedTextPreprocessor()
    calculator = MultiMethodSimilarityCalculator(preprocessor)
    visualizer = PlagiarismVisualizer()
    
    # Documents par défaut
    default_documents = [
        {
            'id': 'doc_1',
            'title': 'IA - Document Original',
            'content': """L'intelligence artificielle représente une avancée majeure en informatique.
Elle permet aux machines d'exécuter des tâches complexes autrefois réservées aux humains.
Les algorithmes de machine learning analysent des données massives pour identifier des patterns.
Le deep learning utilise des réseaux neuronaux multicouches pour résoudre des problèmes difficiles."""
        },
        {
            'id': 'doc_2',
            'title': 'IA - Version Plagiée',
            'content': """L'intelligence artificielle constitue une avancée majeure en informatique.
Elle permet aux systèmes d'exécuter des tâches complexes autrefois réservées aux humains.
Les algorithmes de machine learning analysent des données massives pour détecter des patterns.
Le deep learning utilise des réseaux neuronaux multicouches pour résoudre des problèmes difficiles."""
        },
        {
            'id': 'doc_3',
            'title': 'Biologie Moléculaire',
            'content': """La biologie moléculaire étudie les mécanismes de réplication de l'ADN.
La transcription de l'ADN en ARN messager est une étape cruciale de l'expression génétique.
Les protéines sont synthétisées par les ribosomes à partir de l'information génétique.
La régulation épigénétique influence l'expression des gènes sans modifier la séquence d'ADN."""
        },
        {
            'id': 'doc_4',
            'title': 'IA - Réécriture Partielle',
            'content': """L'intelligence artificielle constitue un progrès significatif dans le domaine informatique.
Les systèmes intelligents peuvent réaliser des opérations complexes précédemment effectuées par des personnes.
Les méthodes d'apprentissage automatique examinent de vastes ensembles de données pour repérer des tendances.
L'apprentissage profond emploie des architectures neuronales stratifiées pour traiter des questions complexes."""
        }
    ]
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Documents", 
        "📊 Résultats", 
        "📈 Visualisations",
        "🔍 Analyse Détail",
        "📁 Export"
    ])
    
    # Onglet 1 : Documents
    with tab1:
        st.markdown('<h3 class="sub-header">📝 Documents à Analyser</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 **Document Source (Original)**")
            
            # Sélection d'exemple
            if example == "Plagiat Évident (IA)":
                default_text1 = default_documents[0]['content']
                default_text2 = default_documents[1]['content']
            elif example == "Plagiat Modéré (IA)":
                default_text1 = default_documents[0]['content']
                default_text2 = default_documents[3]['content']
            elif example == "Non Plagiat (Différents domaines)":
                default_text1 = default_documents[0]['content']
                default_text2 = default_documents[2]['content']
            else:
                default_text1 = default_documents[0]['content']
                default_text2 = default_documents[1]['content']
            
            doc1_content = st.text_area(
                "Document source",
                value=default_text1,
                height=250,
                placeholder="Collez ou écrivez votre document source ici...",
                key="doc1_area"
            )
            
            with st.expander("📊 Statistiques du document 1"):
                if doc1_content:
                    words = len(doc1_content.split())
                    chars = len(doc1_content)
                    sentences = len(re.split(r'[.!?]+', doc1_content))
                    st.metric("Mots", words)
                    st.metric("Caractères", chars)
                    st.metric("Phrases", sentences)
        
        with col2:
            st.markdown("#### 📝 **Document à Vérifier**")
            
            doc2_content = st.text_area(
                "Document à vérifier",
                value=default_text2,
                height=250,
                placeholder="Collez ou écrivez le document suspect ici...",
                key="doc2_area"
            )
            
            with st.expander("📊 Statistiques du document 2"):
                if doc2_content:
                    words = len(doc2_content.split())
                    chars = len(doc2_content)
                    sentences = len(re.split(r'[.!?]+', doc2_content))
                    st.metric("Mots", words)
                    st.metric("Caractères", chars)
                    st.metric("Phrases", sentences)
        
        # Bouton d'analyse
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 **Lancer l'Analyse Complète**", type="primary", use_container_width=True):
                if doc1_content and doc2_content:
                    with st.spinner("🔄 Analyse en cours... Calcul des similarités multi-méthodes"):
                        # Créer les documents pour l'analyse
                        current_documents = [
                            {'id': 'doc_1', 'title': 'Document Source', 'content': doc1_content},
                            {'id': 'doc_2', 'title': 'Document à Vérifier', 'content': doc2_content},
                            default_documents[2],
                            default_documents[3]
                        ]
                        
                        # Analyser toutes les paires
                        all_results = analyze_all_pairs(current_documents, calculator, preprocessor)
                        
                        # Trouver le résultat principal
                        main_result = None
                        for result in all_results:
                            if result['pair_id'] == 'doc_1_doc_2':
                                main_result = result
                                break
                        
                        if main_result is None and all_results:
                            main_result = all_results[0]
                        
                        # Sauvegarder dans session state
                        st.session_state.all_results = all_results
                        st.session_state.main_result = main_result
                        st.session_state.documents = current_documents
                        
                        st.success("✅ Analyse terminée avec succès!")
                        
                        # Afficher les métriques immédiates
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("TF-IDF", f"{main_result['tfidf_cosine']:.2%}")
                        
                        with col2:
                            st.metric("Jaccard", f"{main_result['jaccard']:.2%}")
                        
                        with col3:
                            st.metric("N-gram (3)", f"{main_result['ngram_3']:.2%}")
                        
                        with col4:
                            delta_color = "inverse"
                            if main_result['combined_score'] >= high_threshold:
                                delta_color = "off"
                            st.metric(
                                "Score Combiné", 
                                f"{main_result['combined_score']:.2%}",
                                delta=main_result['plagiarism_decision'],
                                delta_color=delta_color
                            )
                else:
                    st.error("❌ Veuillez entrer les deux documents")
    
    # Onglet 2 : Résultats
    with tab2:
        st.markdown('<h3 class="sub-header">📊 Résultats Détaillés</h3>', unsafe_allow_html=True)
        
        if 'main_result' in st.session_state:
            result = st.session_state.main_result
            
            # Métriques principales
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 **Scores par Méthode**")
                
                df_methods = pd.DataFrame({
                    'Méthode': ['TF-IDF Cosine', 'Jaccard', 'N-gram (2)', 'N-gram (3)', 'N-gram (4)', 'LCS', 'Edit Distance'],
                    'Score': [
                        result['tfidf_cosine'],
                        result['jaccard'],
                        result['ngram_2'],
                        result['ngram_3'],
                        result['ngram_4'],
                        result['lcs'],
                        result['edit_distance']
                    ]
                })
                
                fig_scores = px.bar(
                    df_methods, 
                    x='Méthode', 
                    y='Score',
                    color='Score',
                    color_continuous_scale='RdYlGn_r',
                    text_auto='.3f',
                    height=400
                )
                
                fig_scores.update_traces(
                    textfont_size=12,
                    textangle=0,
                    textposition="outside",
                    cliponaxis=False
                )
                
                fig_scores.update_layout(
                    xaxis_title="",
                    yaxis_title="Score",
                    yaxis_range=[0, 1]
                )
                
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 **Décision de Plagiat**")
                
                # Jauge de décision
                fig_gauge = visualizer.create_combined_score_gauge(
                    result['combined_score'],
                    result['color']
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Affichage de la décision
                decision_class = ""
                if result['combined_score'] >= high_threshold:
                    decision_class = "plagiarism-high"
                elif result['combined_score'] >= moderate_threshold:
                    decision_class = "plagiarism-moderate"
                else:
                    decision_class = "plagiarism-low"
                
                st.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <h3 class="{decision_class}" style="font-size: 1.5rem; padding: 1rem;">
                        {result['plagiarism_decision']}
                    </h3>
                    <p style="color: #6B7280; font-size: 1rem;">
                        {result['plagiarism_level']}<br>
                        <strong>Niveau de confiance:</strong> {result['confidence']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Statistiques détaillées
            st.markdown("#### 📊 **Statistiques Détaillées**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Segments Similaires",
                    result['similar_segments_count'],
                    help="Nombre de segments avec similarité > 70%"
                )
            
            with col2:
                st.metric(
                    "Mots Communs",
                    result['common_words_count'],
                    help="Mots apparaissant au moins 2 fois dans les deux documents"
                )
            
            with col3:
                st.metric(
                    "Longueur Normalisée",
                    f"{result['normalized_length']:.2f}",
                    help="Ratio entre les longueurs des documents"
                )
            
            with col4:
                st.metric(
                    "Score Maximum",
                    f"{max([result['tfidf_cosine'], result['jaccard'], result['lcs']]):.3f}",
                    help="Score le plus élevé parmi toutes les méthodes"
                )
            
            # Tableau des scores complets
            st.markdown("#### 📋 **Tableau des Scores Complets**")
            
            scores_data = {
                'Méthode': [
                    'TF-IDF Cosine', 'Jaccard', 'N-gram (2)', 'N-gram (3)', 
                    'N-gram (4)', 'LCS', 'Edit Distance', 'SCORE COMBINÉ'
                ],
                'Score': [
                    result['tfidf_cosine'], result['jaccard'], result['ngram_2'],
                    result['ngram_3'], result['ngram_4'], result['lcs'],
                    result['edit_distance'], result['combined_score']
                ],
                'Description': [
                    'Similarité sémantique basée sur TF-IDF',
                    'Chevauchement lexical entre documents',
                    'Similarité des séquences de 2 mots',
                    'Similarité des séquences de 3 mots',
                    'Similarité des séquences de 4 mots',
                    'Longest Common Subsequence',
                    'Distance de Levenshtein normalisée',
                    'Score final pondéré (décision)'
                ]
            }
            
            df_scores = pd.DataFrame(scores_data)
            df_scores['Score'] = df_scores['Score'].apply(lambda x: f"{x:.3f}")
            st.dataframe(df_scores, use_container_width=True, hide_index=True)
        
        else:
            st.info("ℹ️ Veuillez d'abord effectuer une analyse dans l'onglet '📝 Documents'")
    
    # Onglet 3 : Visualisations
    with tab3:
        st.markdown('<h3 class="sub-header">📈 Visualisations Interactives</h3>', unsafe_allow_html=True)
        
        if 'main_result' in st.session_state and 'all_results' in st.session_state:
            result = st.session_state.main_result
            all_results = st.session_state.all_results
            
            # Row 1: Radar et Heatmap
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 **Radar des Similarités**")
                fig_radar = visualizer.create_similarity_radar(
                    result,
                    ['Document Source', 'Document Vérifié']
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔥 **Matrice de Similarité**")
                fig_heatmap = visualizer.create_specific_heatmap(all_results)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Row 2: Comparaison et Méthodes
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 **Comparaison des Paires**")
                fig_comparison = visualizer.create_specific_pairs_comparison(all_results)
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔧 **5 Méthodes + Score Combiné**")
                fig_methods = visualizer.create_methods_comparison_with_combined(result)
                st.plotly_chart(fig_methods, use_container_width=True)
            
            # Dashboard complet
            st.markdown("#### 📋 **Dashboard Complet**")
            
            fig_dashboard = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Radar des Similarités', 
                    'Matrice de Similarité',
                    'Comparaison des Paires', 
                    '5 Méthodes + Score Combiné',
                    'Jauge du Score', 
                    'Tableau des Scores'
                ),
                specs=[
                    [{'type': 'polar'}, {'type': 'heatmap'}],
                    [{'type': 'xy'}, {'type': 'xy'}],
                    [{'type': 'indicator'}, {'type': 'table'}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.15
            )
            
            # Ajouter chaque visualisation
            fig_radar_data = visualizer.create_similarity_radar(result, ['Doc1', 'Doc2'])
            for trace in fig_radar_data.data:
                fig_dashboard.add_trace(trace, row=1, col=1)
            
            fig_heatmap_data = visualizer.create_specific_heatmap(all_results)
            for trace in fig_heatmap_data.data:
                fig_dashboard.add_trace(trace, row=1, col=2)
            
            fig_comparison_data = visualizer.create_specific_pairs_comparison(all_results)
            for trace in fig_comparison_data.data:
                fig_dashboard.add_trace(trace, row=2, col=1)
            
            fig_methods_data = visualizer.create_methods_comparison_with_combined(result)
            for trace in fig_methods_data.data:
                fig_dashboard.add_trace(trace, row=2, col=2)
            
            fig_gauge_data = visualizer.create_combined_score_gauge(result['combined_score'], result['color'])
            for trace in fig_gauge_data.data:
                fig_dashboard.add_trace(trace, row=3, col=1)
            
            # Tableau des scores
            scores_table = go.Table(
                header=dict(
                    values=['<b>MÉTHODE</b>', '<b>SCORE</b>', '<b>INTERPRÉTATION</b>'],
                    fill_color='#1E3A8A',
                    align='center',
                    font=dict(color='white', size=13),
                    height=40
                ),
                cells=dict(
                    values=[
                        ['TF-IDF Cosine', 'Jaccard', 'N-gram (2)', 'LCS', 'Edit Distance', '<b>SCORE COMBINÉ</b>'],
                        [f"{result['tfidf_cosine']:.3f}", 
                         f"{result['jaccard']:.3f}", 
                         f"{result['ngram_2']:.3f}", 
                         f"{result['lcs']:.3f}", 
                         f"{result['edit_distance']:.3f}",
                         f"<b>{result['combined_score']:.3f}</b>"],
                        ['Similarité sémantique', 
                         'Chevauchement lexical', 
                         'Séquences de 2 mots', 
                         'Sous-séquences communes', 
                         'Distance d\'édition',
                         f"<b>{result['plagiarism_decision']}</b>"]
                    ],
                    fill_color=['lightgray', 'white', 'lightgray'],
                    align=['center', 'center', 'left'],
                    font=dict(size=12),
                    height=35
                )
            )
            
            fig_dashboard.add_trace(scores_table, row=3, col=2)
            
            fig_dashboard.update_layout(
                height=1200,
                showlegend=False,
                title_text="PLAGGRAPH-EXPLAIN - DASHBOARD COMPLET",
                title_font_size=22,
                title_font_color='#1E3A8A',
                title_x=0.5,
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig_dashboard, use_container_width=True)
        
        else:
            st.info("ℹ️ Veuillez d'abord effectuer une analyse dans l'onglet '📝 Documents'")
    
    # Onglet 4 : Analyse Détail
    with tab4:
        st.markdown('<h3 class="sub-header">🔍 Analyse Détaillée</h3>', unsafe_allow_html=True)
        
        if 'main_result' in st.session_state:
            result = st.session_state.main_result
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 **Segments Similaires Identifiés**")
                
                if result['similar_segments_count'] > 0:
                    for i, segment in enumerate(result['similar_segments'], 1):
                        with st.expander(f"Segment {i} - Similarité: {segment['similarity']:.1%}", expanded=i==1):
                            st.markdown("**📄 Document Source:**")
                            st.info(segment['segment1'])
                            
                            st.markdown("**📝 Document Vérifié:**")
                            st.warning(segment['segment2'])
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Longueur 1", f"{len(segment['segment1'])} caractères")
                            with col_b:
                                st.metric("Longueur 2", f"{len(segment['segment2'])} caractères")
                else:
                    st.info("Aucun segment fortement similaire (similarité > 70%) n'a été identifié.")
            
            with col2:
                st.markdown("#### 🔤 **Mots Communs Fréquents**")
                
                if result['common_words_count'] > 0:
                    words_data = []
                    for word, data in result['top_common_words']:
                        if isinstance(data, dict):
                            words_data.append({
                                'Mot': word,
                                'Fréq. Doc1': data.get('doc1_freq', 0),
                                'Fréq. Doc2': data.get('doc2_freq', 0),
                                'Total': data.get('total', 0)
                            })
                    
                    if words_data:
                        df_words = pd.DataFrame(words_data)
                        
                        fig_words = px.bar(
                            df_words.head(10),
                            x='Mot',
                            y='Total',
                            color='Total',
                            color_continuous_scale='Blues',
                            title='Top 10 Mots Communs (fréquence totale)',
                            height=400
                        )
                        
                        fig_words.update_layout(
                            xaxis_title="",
                            yaxis_title="Fréquence Totale",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_words, use_container_width=True)
                        
                        with st.expander("📋 Voir tous les mots communs"):
                            st.dataframe(df_words, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun mot commun fréquent n'a été identifié.")
                else:
                    st.info("Aucun mot commun (apparaissant au moins 2 fois) n'a été identifié.")
            
            # Analyse comparative
            st.markdown("#### 📊 **Analyse Comparative**")
            
            if 'all_results' in st.session_state:
                all_results = st.session_state.all_results
                
                specific_pairs = ['doc_1_doc_2', 'doc_1_doc_3', 'doc_1_doc_4']
                filtered_results = [r for r in all_results if r['pair_id'] in specific_pairs]
                
                if filtered_results:
                    comparison_data = []
                    for r in filtered_results:
                        comparison_data.append({
                            'Paire': f"{r['doc1']}↔{r['doc2']}",
                            'TF-IDF': r['tfidf_cosine'],
                            'Jaccard': r['jaccard'],
                            'N-gram (2)': r['ngram_2'],
                            'LCS': r['lcs'],
                            'Edit Dist': r['edit_distance'],
                            'Score Comb.': r['combined_score'],
                            'Décision': r['plagiarism_decision']
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 📋 Tableau Comparatif")
                        st.dataframe(
                            df_comparison.style.format({
                                'TF-IDF': '{:.3f}',
                                'Jaccard': '{:.3f}',
                                'N-gram (2)': '{:.3f}',
                                'LCS': '{:.3f}',
                                'Edit Dist': '{:.3f}',
                                'Score Comb.': '{:.3f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.markdown("##### 📈 Statistiques des Scores")
                        
                        scores_combined = df_comparison['Score Comb.'].values
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            st.metric("Moyenne", f"{np.mean(scores_combined):.3f}")
                            st.metric("Maximum", f"{np.max(scores_combined):.3f}")
                        
                        with metrics_col2:
                            st.metric("Médiane", f"{np.median(scores_combined):.3f}")
                            st.metric("Minimum", f"{np.min(scores_combined):.3f}")
                        
                        st.metric("Écart-type", f"{np.std(scores_combined):.3f}")
        
        else:
            st.info("ℹ️ Veuillez d'abord effectuer une analyse dans l'onglet '📝 Documents'")
    
    # Onglet 5 : Export
    with tab5:
        st.markdown('<h3 class="sub-header">📁 Export des Résultats</h3>', unsafe_allow_html=True)
        
        if 'main_result' in st.session_state:
            result = st.session_state.main_result
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📄 **Rapport Texte**")
                
                text_report = f"""
PLAGGRAPH-EXPLAIN - RAPPORT D'ANALYSE
=======================================
Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Paire analysée: {result['doc1']} ↔ {result['doc2']}
Titres: {result['doc1_title']} | {result['doc2_title']}

SCORES DE SIMILARITÉ:
----------------------
TF-IDF Cosine:    {result['tfidf_cosine']:.4f}
Jaccard:          {result['jaccard']:.4f}
N-gram (2):       {result['ngram_2']:.4f}
N-gram (3):       {result['ngram_3']:.4f}
N-gram (4):       {result['ngram_4']:.4f}
LCS:              {result['lcs']:.4f}
Edit Distance:    {result['edit_distance']:.4f}
Score Combiné:    {result['combined_score']:.4f}

DÉCISION:
----------
Verdict:          {result['plagiarism_decision']}
Niveau:           {result['plagiarism_level']}
Confiance:        {result['confidence']}

STATISTIQUES:
-------------
Segments similaires: {result['similar_segments_count']}
Mots communs:        {result['common_words_count']}
Longueur normalisée: {result['normalized_length']:.4f}

SYSTÈME:
--------
PlagGraph-Explain v2.0
Analyse multi-méthodes avec 5 algorithmes
                """
                
                st.download_button(
                    label="📥 Télécharger rapport TXT",
                    data=text_report,
                    file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🗂️ **Rapport JSON**")
                
                json_report = generate_json_report(result)
                
                st.download_button(
                    label="📥 Télécharger rapport JSON",
                    data=json_report,
                    file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("#### 📊 **Données CSV**")
                
                if 'all_results' in st.session_state:
                    all_results = st.session_state.all_results
                    
                    csv_data = []
                    for r in all_results:
                        csv_data.append({
                            'pair_id': r['pair_id'],
                            'doc1': r['doc1'],
                            'doc2': r['doc2'],
                            'tfidf_cosine': r['tfidf_cosine'],
                            'jaccard': r['jaccard'],
                            'ngram_2': r['ngram_2'],
                            'ngram_3': r['ngram_3'],
                            'ngram_4': r['ngram_4'],
                            'lcs': r['lcs'],
                            'edit_distance': r['edit_distance'],
                            'combined_score': r['combined_score'],
                            'decision': r['plagiarism_decision']
                        })
                    
                    df_csv = pd.DataFrame(csv_data)
                    csv_string = df_csv.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Télécharger données CSV",
                        data=csv_string,
                        file_name=f"plagiarism_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Visualisations exportables
            st.markdown("#### 📈 **Export des Visualisations**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'main_result' in st.session_state:
                    # Exporter le radar chart
                    fig_radar = visualizer.create_similarity_radar(
                        st.session_state.main_result,
                        ['Document Source', 'Document Vérifié']
                    )
                    
                    # Convertir en HTML
                    radar_html = fig_radar.to_html(full_html=False, include_plotlyjs='cdn')
                    
                    st.download_button(
                        label="📥 Exporter Radar Chart (HTML)",
                        data=radar_html,
                        file_name=f"radar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            
            with col2:
                if 'main_result' in st.session_state:
                    # Exporter le dashboard
                    fig_gauge = visualizer.create_combined_score_gauge(
                        st.session_state.main_result['combined_score'],
                        st.session_state.main_result['color']
                    )
                    
                    gauge_html = fig_gauge.to_html(full_html=False, include_plotlyjs='cdn')
                    
                    st.download_button(
                        label="📥 Exporter Jauge (HTML)",
                        data=gauge_html,
                        file_name=f"gauge_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            
            # Rapport complet
            st.markdown("#### 📋 **Rapport Complet**")
            
            full_report = f"""
# RAPPORT COMPLET PLAGGRAPH-EXPLAIN

## 📋 Informations Générales
- **Date d'analyse**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
- **Paire analysée**: {result['doc1']} ↔ {result['doc2']}
- **Système**: PlagGraph-Explain v2.0

## 📊 Scores Détailés
| Méthode | Score | Description |
|---------|-------|-------------|
| TF-IDF Cosine | {result['tfidf_cosine']:.4f} | Similarité sémantique |
| Jaccard | {result['jaccard']:.4f} | Chevauchement lexical |
| N-gram (2) | {result['ngram_2']:.4f} | Séquences de 2 mots |
| N-gram (3) | {result['ngram_3']:.4f} | Séquences de 3 mots |
| N-gram (4) | {result['ngram_4']:.4f} | Séquences de 4 mots |
| LCS | {result['lcs']:.4f} | Sous-séquences communes |
| Edit Distance | {result['edit_distance']:.4f} | Distance de Levenshtein |
| **Score Combiné** | **{result['combined_score']:.4f}** | **Score final pondéré** |

## 🎯 Décision Finale
**{result['plagiarism_decision']}**
- Niveau: {result['plagiarism_level']}
- Confiance: {result['confidence']}

## 📈 Statistiques
- Segments similaires identifiés: {result['similar_segments_count']}
- Mots communs fréquents: {result['common_words_count']}
- Ratio de longueur: {result['normalized_length']:.4f}

## 🔧 Méthodologie
Le système utilise 5 méthodes de similarité combinées avec des poids:
- TF-IDF Cosine (30%)
- Jaccard (15%)
- N-gram (2, 3, 4) (35%)
- LCS (10%)
- Edit Distance (10%)

Seuils de décision:
- ≥ 0.7: Plagiat Élevé
- ≥ 0.5: Similarité Modérée
- < 0.5: Non Plagiat

## 📞 Support
Pour toute question ou support technique, contactez l'équipe PlagGraph-Explain.
            """
            
            st.download_button(
                label="📥 Télécharger Rapport Complet (Markdown)",
                data=full_report,
                file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        else:
            st.info("ℹ️ Veuillez d'abord effectuer une analyse pour pouvoir exporter les résultats")
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">
            <p>🔍 <strong>PlagGraph-Explain v2.0</strong> - Système de détection de plagiat explicable avec IA</p>
            <p>© 2024 - Combinaison de 5 méthodes de similarité avec visualisations interactives</p>
            <p style="margin-top: 0.5rem;">
                <span style="background: #F3F4F6; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem;">
                    ✅ TF-IDF Cosine
                </span>
                <span style="background: #F3F4F6; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem;">
                    ✅ Jaccard
                </span>
                <span style="background: #F3F4F6; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem;">
                    ✅ N-gram
                </span>
                <span style="background: #F3F4F6; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem;">
                    ✅ LCS
                </span>
                <span style="background: #F3F4F6; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem;">
                    ✅ Edit Distance
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# EXÉCUTION
# ============================================================================
if __name__ == "__main__":
    main()
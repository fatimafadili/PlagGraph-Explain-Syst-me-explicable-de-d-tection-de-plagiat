# PlagGraph-Explain-Syst-me-explicable-de-d-tection-de-plagiat
📖 Table des Matières
🌟 Aperçu

🎯 Fonctionnalités

📊 Méthodes de Similarité

🚀 Installation Rapide

💻 Utilisation

📁 Structure du Projet

🔧 Configuration Avancée

📈 Résultats et Visualisations

📚 Documentation Technique

🤝 Contribution

📄 Licence

👥 Auteurs

🌟 Étoiles

🌟 Aperçu
PlagGraph-Explain est un système avancé de détection de plagiat qui combine 5 algorithmes de similarité avec des visualisations interactives et des explications détaillées. Conçu pour la transparence et l'explicabilité, le système permet de comprendre pourquoi un document est considéré comme plagié, pas seulement si il l'est.

🔑 Points Forts
✅ Multi-méthodes : Combinaison de 5 algorithmes de similarité

✅ Explicabilité : Visualisations interactives pour comprendre les décisions

✅ Interface moderne : Application Streamlit avec design professionnel

✅ Export complet : Rapports TXT, JSON, CSV, HTML

✅ Documents de test : Exemples avec différents niveaux de plagiat

🎯 Fonctionnalités
🎨 Interface Utilisateur
Design responsive avec CSS personnalisé

Navigation par onglets intuitive

Sidebar configurable avec paramètres ajustables

Animations et transitions fluides

Thème moderne avec gradient et ombres

📊 Analyse Avancée
5 méthodes de similarité combinées

Score pondéré avec seuils configurables

Détection de segments similaires

Analyse de mots communs fréquents

Statistiques détaillées par document

📈 Visualisations Interactives
Radar Chart : Comparaison des 5 méthodes

Heatmap : Matrice de similarité entre documents

Graphiques à barres : Scores détaillés

Jauge : Score combiné avec seuils colorés

Dashboard complet : Toutes les visualisations intégrées

📁 Export et Rapports
Rapport texte détaillé (.txt)

Données structurées (.json)

Tableaux exportables (.csv)

Visualisations HTML interactives

Rapport complet Markdown

📊 Méthodes de Similarité
Méthode	Algorithme	Poids	Description
TF-IDF Cosine	Cosine Similarity	30%	Similarité sémantique basée sur la fréquence des termes
Jaccard	Jaccard Index	15%	Chevauchement lexical entre ensembles de mots
N-gram (2,3,4)	N-gram Overlap	35%	Similarité des séquences de 2, 3 et 4 mots
LCS	Longest Common Subsequence	10%	Sous-séquences communes les plus longues
Edit Distance	Levenshtein Distance	10%	Distance d'édition normalisée
🎯 Seuils de Décision
≥ 0.7 : 🔴 PLAGIAT ÉLEVÉ - Action immédiate requise

≥ 0.5 : 🟡 SIMILARITÉ MODÉRÉE - Vérification recommandée

< 0.5 : 🟢 NON PLAGIAT - Aucune action nécessaire

🚀 Installation Rapide
Prérequis
Python 3.8 ou supérieur

pip (gestionnaire de paquets Python)

500MB d'espace disque libre

📦 Installation en 3 Étapes
Cloner le dépôt

bash
git clone https://github.com/votre-username/plaggraph-explain.git
cd plaggraph-explain
Créer un environnement virtuel (recommandé)

bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Installer les dépendances

bash
pip install -r requirements.txt
🛠️ Vérification de l'installation
bash
python -c "import streamlit; import plotly; print('✅ Installation réussie!')"
💻 Utilisation
🖥️ Lancer l'Application
bash
streamlit run app.py
L'application sera accessible à l'adresse : http://localhost:8501

📝 Guide d'Utilisation Étape par Étape
Étape 1 : Sélection des Documents
Onglet "Documents"

Choisissez un exemple prédéfini :

Plagiat Évident (IA) : Documents presque identiques

Plagiat Modéré (IA) : Documents partiellement similaires

Non Plagiat : Documents de domaines différents

Ou collez vos propres documents dans les zones de texte

Étape 2 : Configuration
Sidebar → Paramètres d'Analyse

Ajustez les seuils de décision si nécessaire

Sélectionnez les méthodes à utiliser

Cliquez sur "Lancer l'Analyse Complète"

Étape 3 : Analyse des Résultats
Onglet "Résultats" : Scores détaillés et décision

Onglet "Visualisations" : Graphiques interactifs

Onglet "Analyse Détail" : Segments similaires et mots communs

Onglet "Export" : Téléchargement des rapports

🎮 Fonctionnalités Clavier
Ctrl + R : Rafraîchir la page

Ctrl + S : Sauvegarder les paramètres

Ctrl + E : Exporter les résultats

Esc : Retour à l'accueil

📁 Structure du Projet
text
plaggraph-explain/
│
├── 📁 src/                           # Code source principal
│   ├── __init__.py                  # Package initialisation
│   ├── preprocessor.py              # Prétraitement du texte
│   ├── similarity.py                # Calcul des similarités
│   ├── explainer.py                 # Explications LIME/SHAP
│   └── visualizer.py                # Visualisations Plotly
│
├── 📁 data/                         # Données et documents
│   ├── examples/                    # Exemples prédéfinis
│   │   ├── plagiarism_high.txt     # Plagiat évident
│   │   ├── plagiarism_moderate.txt # Plagiat modéré
│   │   └── no_plagiarism.txt       # Non plagiat
│   └── test_documents.json         # Documents de test structurés
│
├── 📁 notebooks/                    # Notebooks d'analyse
│   ├── 01_data_exploration.ipynb   # Exploration des données
│   ├── 02_similarity_analysis.ipynb # Analyse des similarités
│   └── 03_visualizations.ipynb     # Création des visualisations
│
├── 📁 reports/                      # Rapports générés
│   ├── templates/                  # Templates de rapports
│   └── examples/                   # Exemples de rapports
│
├── 📁 tests/                        # Tests unitaires
│   ├── test_preprocessor.py        # Tests prétraitement
│   ├── test_similarity.py          # Tests similarités
│   └── test_visualizer.py          # Tests visualisations
│
├── 📁 docs/                         # Documentation
│   ├── api/                        # Documentation API
│   ├── user_guide/                 # Guide utilisateur
│   └── technical/                  # Documentation technique
│
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── Dockerfile                     # Configuration Docker
├── docker-compose.yml             # Orchestration Docker
├── .env.example                   Variables d'environnement
├── .gitignore                    # Fichiers ignorés Git
├── LICENSE                       # Licence MIT
└── README.md                     # Ce fichier
🔧 Configuration Avancée
⚙️ Variables d'Environnement
Créez un fichier .env à la racine :

env
# Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
DEBUG_MODE=False

# Seuils par défaut
PLAGIARISM_HIGH_THRESHOLD=0.7
PLAGIARISM_MODERATE_THRESHOLD=0.5

# Poids des méthodes
TFIDF_WEIGHT=0.3
JACCARD_WEIGHT=0.15
NGRAM_WEIGHT=0.35
LCS_WEIGHT=0.1
EDIT_WEIGHT=0.1


📚 Documentation Technique
🧠 Architecture du Système
graph TD
    A[Document Source] --> B{Prétraitement};
    C[Document à Vérifier] --> B;
    B --> D[Calcul Similarités Multi-Méthodes];
    D --> E[TF-IDF Cosine];
    D --> F[Jaccard];
    D --> G[N-gram];
    D --> H[LCS];
    D --> I[Edit Distance];
    E --> J{Combinaison Pondérée};
    F --> J;
    G --> J;
    H --> J;
    I --> J;
    J --> K[Décision de Plagiat];
    K --> L[Visualisations];
    K --> M[Rapports];
    K --> N[Explications];
🔬 Algorithmes Implémentés
TF-IDF Cosine Similarity
python
def calculate_tfidf_cosine(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity
Longest Common Subsequence (LCS)
python
def calculate_lcs_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    m, n = len(words1), len(words2)
    L = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        for j in range(n+1):
            if i==0 or j==0:
                L[i][j] = 0
            elif words1[i-1] == words2[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    
    return L[m][n] / ((m + n) / 2)
📊 Métriques de Performance
python
# Calcul de précision, rappel et F1-score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Matrice de confusion
confusion_matrix = [[tn, fp], [fn, tp]]
🔍 Tests Unitaires
bash
# Exécuter tous les tests
python -m pytest tests/ -v

# Tests spécifiques
python -m pytest tests/test_similarity.py -v
python -m pytest tests/test_preprocessor.py -v

# Avec couverture de code
python -m pytest tests/ --cov=src --cov-report=html
🤝 Contribution
🏗️ Comment Contribuer
Fork le projet

Clone votre fork

Créez une branche (git checkout -b feature/AmazingFeature)

Commitez vos changements (git commit -m 'Add AmazingFeature')

Push vers la branche (git push origin feature/AmazingFeature)

Ouvrez une Pull Request

📋 Bonnes Pratiques de Code
Utilisez des noms de variables descriptifs

Commentez votre code (docstrings pour les fonctions)

Suivez PEP 8 (guide de style Python)

Écrivez des tests unitaires pour les nouvelles fonctionnalités

Mettez à jour la documentation correspondante

🐛 Rapport de Bugs
Utilisez les Issues GitHub avec le modèle suivant :

markdown
## Description du Bug
[Description claire et concise]

## Étapes pour reproduire
1. Aller à '...'
2. Cliquer sur '....'
3. Scroller jusqu'à '....'
4. Voir l'erreur

## Comportement attendu
[Description de ce qui devrait se passer]

## Captures d'écran
[Si applicable, ajoutez des captures d'écran]

## Environnement
- OS: [ex: Windows 10, macOS 12.0]
- Navigateur: [ex: Chrome 96, Safari 15]
- Version Python: [ex: 3.9.7]

## Informations supplémentaires
[Ajoutez tout autre contexte sur le problème]
🌟 Fonctionnalités Planifiées
Intégration SHAP pour l'explicabilité

Support multilingue (anglais, espagnol, allemand)

API REST pour intégration externe

Base de données pour historique des analyses

Plugins pour extensions tierces

Analyse en temps réel avec WebSockets

Intégration LMS (Moodle, Canvas)

Mobile App (React Native)

📄 Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

text
MIT License

Copyright (c) 2024 PlagGraph-Explain Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

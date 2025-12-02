# data/sample_documents.py
def get_sample_documents():
    """Retourne des documents d'exemple pour tester le système"""
    
    documents = [
        # Document 1 - Original
        {
            "id": "doc_1",
            "title": "Deep Learning pour la Reconnaissance d'Images",
            "content": """
            Les réseaux de neurones convolutifs (CNN) représentent l'état de l'art en reconnaissance d'images. 
            Ces modèles deep learning utilisent des couches convolutionnelles pour extraire des caractéristiques 
            hiérarchiques des images. La performance des CNN a dépassé les méthodes traditionnelles de computer 
            vision. Des architectures comme ResNet, VGG et Inception ont établi de nouveaux records sur des 
            benchmarks comme ImageNet. L'apprentissage par transfert permet de fine-tuner ces modèles pré-entraînés 
            pour des tâches spécifiques avec peu de données. Les CNN sont très efficaces pour la classification d'images.
            """
        },
        
        # Document 2 - Plagiat partiel (85% similarité)
        {
            "id": "doc_2", 
            "title": "Avancées en Vision par Ordinateur",
            "content": """
            Les réseaux neuronaux convolutifs (CNN) constituent la référence en reconnaissance d'images. 
            Ces architectures deep learning emploient des couches convolutionnelles pour extraire des traits 
            hiérarchiques depuis les images. Les performances des CNN ont surpassé les approches classiques 
            en vision artificielle. Des modèles tels que ResNet, VGG et Inception ont établi des records sur 
            des benchmarks comme ImageNet. L'apprentissage par transfert permet d'adapter ces modèles pré-entraînés 
            à des applications spécifiques avec peu de données. Les CNN excellent dans la classification d'images.
            """
        },
        
        # Document 3 - Sans rapport
        {
            "id": "doc_3",
            "title": "Traitement du Langage Naturel",
            "content": """
            Les transformers ont révolutionné le traitement du langage naturel. Des modèles comme BERT et GPT utilisent 
            des mécanismes d'attention pour capturer les dépendances contextuelles dans le texte. Ces architectures 
            permettent d'obtenir des performances state-of-the-art sur des tâches comme la classification de texte, 
            la réponse à des questions et la génération de langage.
            """
        }
    ]
    
    return documents
# 🌟 Projet Final : Classification d’Images DTD (CNN vs. Apprentissage Traditionnel)

Ce projet compare deux pipelines de classification d'images pour le jeu de données de textures DTD (Describable Textures Dataset) :
1. **Pipeline Traditionnel (Transfer Learning)** : Extraction de caractéristiques (VGG16, InceptionV3, ResNet50) + Modèles classiques (SVM, k-NN, Arbre de Décision, Naïve Bayes).
2. **Pipeline CNN Personnalisé** : Entraînement d'un CNN *from scratch*.

## 🚀 Exécution du Projet

1. **Installation** : `pip install -r requirements.txt`
2. **Données** : Placer les images DTD dans `data/raw/dtd_images/`.
3. **Phases d'Exécution** :
    * **Préparation des Données** : Exécuter `python src/data_loader.py`
    * **Phase 1 (Traditionnelle)** : Exécuter `python src/feature_extractor.py` puis `python src/train_classical.py`
    * **Phase 2 (CNN)** : Exécuter `python src/train_cnn.py`
    * **Évaluation Finale** : Exécuter `python src/evaluate_models.py`
    * **Analyse** : Ouvrir `4_analysis/4.1_analysis_report.ipynb`

## 📂 Structure du Projet

(Copier l'arborescence des dossiers ici.)





# DTD — Classification d'images (Pipeline traditionnel vs CNN)

## Prérequis
- Python 3.9+
- GPU recommandé pour CNN (mais CPU possible)

## Installation
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


# Pour lancer API
uvicorn api.app:app --reload --port 8000

# pour lancer le web ou client
Aller sur dossier web
taper  python -m http.server 8088 (vous pouvez changer de port 8088 si ce port est utiliser deja)
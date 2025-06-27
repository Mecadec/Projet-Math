# Projet Mathématiques - Analyse de Données

Ce projet contient six parties distinctes d'analyse de données, chacune se concentrant sur des aspects différents des statistiques et du machine learning.

## Prérequis

- Python 3.6 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation des dépendances

1. Créez un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Sur Windows
   source venv/bin/activate  # Sur macOS/Linux
   ```

2. Installez les dépendances requises :
   ```bash
   pip install numpy matplotlib pandas scipy scikit-learn seaborn statsmodels openpyxl
   ```

## Structure du projet

```
Projet-Math/
├── data/                      # Dossier contenant les données du projet
│   └── Data_PE_2025-CSI3_CIR3.xlsx
├── Partie_1/                  # Statistiques descriptives de base
│   └── Partie_1.py    
├── Partie_2/                  # Régression linéaire simple
│   ├── Partie_2.py
│   └── graph/                 # Graphiques générés
├── Partie_3/                  # Régression linéaire multiple
│   ├── Partie_3.py
│   └── Graph/                 # Graphiques générés
├── Partie_4/                  # Analyse de variance (ANOVA)
│   └── Partie_4.py
├── Partie_5/                  # Classification hiérarchique
│   └── Partie_5.py
├── Partie_6/                  # Clustering avancé
│   ├── Partie_6.py
│   └── figures/               # Figures générées
└── README.md
```

## Exécution du projet

Chaque partie peut être exécutée indépendamment. Depuis la racine du projet :

```bash
# Pour la Partie 1 (Statistiques descriptives)
python Partie_1/Partie_1.py

# Pour la Partie 2 (Régression linéaire simple)
python Partie_2/Partie_2.py

# Pour la Partie 3 (Régression linéaire multiple)
python Partie_3/Partie_3.py

# Pour la Partie 4 (ANOVA)
python Partie_4/Partie_4.py

# Pour la Partie 5 (Classification hiérarchique)
python Partie_5/Partie_5.py

# Pour la Partie 6 (Clustering avancé)
python Partie_6/Partie_6.py
```

## Dépendances principales

- `numpy` : Calcul numérique
- `pandas` : Manipulation des données
- `matplotlib` et `seaborn` : Visualisation
- `scipy` et `scikit-learn` : Analyses statistiques et machine learning
- `statsmodels` : Modélisation statistique
- `openpyxl` : Lecture des fichiers Excel

## Fichiers générés

- **Partie 2** : 
  - Graphiques : `Partie_2/graph/`
- **Partie 3** : 
  - Graphiques : `Partie_3/Graph/`
  - Résultats : `Partie_3/resultats_regression.csv`
- **Partie 6** : 
  - Figures : `Partie_6/figures/`

## Auteurs

- Armand BEHAREL CIR3
- Gauthier GLOANEC CIR3
- Pol NERISSON CIR3

## Notes

- Assurez-vous que le fichier de données `Data_PE_2025-CSI3_CIR3.xlsx` est bien présent dans le dossier `data/`
- Tous les chemins sont gérés de manière relative pour fonctionner depuis la racine du projet
# Conversion de la Langue des Signes en Texte et Parole

## Description
J'ai réalisé ce projet innovant, développé dans le cadre du programme **M2 MIAGE parcours IA2 (Intelligence Artificielle Appliquée) 2024-2025** dans le module de "Computer Vision" avec **Monsieur Winter**.
Ce projet utilise la vision par ordinateur et l'intelligence artificielle pour convertir des gestes de la langue des signes en texte et en parole. Il prend en charge l'alphabet (vers ce lien [Alphabets](images/alphabet.jpg)), les chiffres (1-9) (vers ce lien [Chiffres](images/Chiffres.png)) et des gestes spécifiques (vers ce lien [Quesques Gestes](images/Gestes.jpeg)). L'application comprend une interface utilisateur interactive et est conçue pour les besoins éducatifs et sociaux.

Lien du dépôt : [Sign Language Recognition](https://github.com/dioprawane/sign_language_computer_vision_IA2)

## Fonctionnalités
- **Reconnaissance de gestes en temps réel** : Conversion des gestes en texte grâce à Mediapipe et un modèle de réseau neuronal.
- **Gestion de texte** : Construction dynamique de mots et phrases à partir des gestes détectés.
- **Synthèse vocale** : Lecture de la phrase construite.
- **Raccourcis clavier** : Contrôles additionnels pour faciliter l'utilisation.

## Structure du Projet
```plaintext
langage_des_signes/
│
├── data/                      # Exemples de données (quelques une issues de datasets)
│   ├── gesture_data/          # Données brutes des gestes collectés
│   ├── processed_data/        # Données prétraitées pour l'entraînement
│
├── datasets/                  # Dossier ignoré (contenant énormément de données) 
│   ├── gesture_data/          # Données brutes des gestes collectés
│   ├── processed_data/        # Données prétraitées pour l'entraînement
│
├── models/
│   ├── gesture_model_V?.h5    # Modèles Keras pour la reconnaissance des gestes
│   ├── label_classes_V?.npy   # Classes des labels (lettres, chiffres, gestes)
│
├── src/
│   ├── config.py              # Configuration globale
│   ├── collect_data.py        # Collecte des données des gestes
│   ├── detection.py           # Détection avancée (Tkinter)
│   ├── gestures.py            # Fonctions pour manipuler les gestes
│   ├── main.py                # Interface utilisateur Tkinter
│   ├── pretreatment.py        # Prétraitement des données
│   ├── recognize.py           # Reconnaissance en temps réel
│   ├── train_model.py         # Entraînement du modèle
│
├── templates/
│   ├── index.html             # Interface HTML de Flask
│
├── utils/
│   ├── choix_model.md         # Choix du modèle
│
├── .gitignore                 # Fichiers à ignorer par Git
├── app.py                     # Application Flask
├── README.md                  # Documentation principale
├── requirements.txt           # Dépendances Python
└── venv/                      # Environnement virtuel Python
```

## Prérequis & Installation
### Prérequis

### Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/dioprawane/sign_language_computer_vision_IA2.git
   cd sign_language_computer_vision_IA2
   ```
2. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sous Windows : venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
### 1. Collecte de données
Pour collecter des gestes :
```bash
python collect_data.py
```
Ajoutez les gestes désirés dans la liste `gestures` pour personnaliser la collecte.

### 2. Prétraitement des données
Préparez les données collectées :
```bash
python pretreatment.py
```
Cela génère des fichiers prêts pour l'entraînement dans `datasets/processed_data/`.

### 3. Entraînement du modèle
Entraînez un modèle de reconnaissance des gestes :
```bash
python train_model.py
```
Le modèle est sauvegardé dans `models/gesture_model_V6.h5`.

### 4. Reconnaissance en temps réel
Lancez la reconnaissance :
```bash
python recognize.py
```

### 5. Interface utilisateur Flask
Lancez le serveur Flask pour utiliser l'application web :
```bash
python app.py
```
Accédez à [http://127.0.0.1:5000](http://127.0.0.1:5000) dans votre navigateur.

## Commandes Clavier
- **`R`** : Réinitialiser
- **`Q`** : Quitter
- **`P`** : Parler
- **`S`** : Supprimer le dernier caractère
- **Espace** : Ajouter un espace
- **`.`** : Ajouter un point

## Structure des Fichiers Importants
- **`app.py`** : Application Flask avec gestion des gestes, phrases et vidéo.
- **`train_model.py`** : Contient la logique d'entraînement avec des couches denses pour la classification.
- **`collect_data.py`** : Utilise Mediapipe pour capturer les landmarks des gestes.
- **`recognize.py`** : Détecte les gestes en temps réel avec OpenCV.
- **`index.html`** : Interface utilisateur web.

## Visualisation des Résultats
L'application offre :
1. **Flux vidéo en temps réel** : Les gestes sont détectés et affichés en direct.
2. **Statut en texte** : Caractères, mots et phrases affichés dynamiquement.
3. **Synthèse vocale** : Lecture des phrases détectées.

## Améliorations Futures
- Ajout d'une base de données pour sauvegarder les sessions utilisateur.
- Détection multi-main et amélioration de la précision des gestes complexes.
- Ajout d'une API REST pour intégrer la reconnaissance dans d'autres applications.

## Contributions
Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request pour des améliorations.

## Auteur
[Serigne Rawane Diop](https://github.com/dioprawane)

## Licence
Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

# Conversion de la Langue des Signes en Texte et Parole

# Sommaire
- [Conversion de la Langue des Signes en Texte et Parole](#conversion-de-la-langue-des-signes-en-texte-et-parole)
- [Sommaire](#sommaire)
  - [Description](#description)
  - [Fonctionnalités](#fonctionnalités)
  - [Structure du Projet](#structure-du-projet)
  - [Prérequis \& Installation](#prérequis--installation)
    - [Prérequis](#prérequis)
    - [Installation](#installation)
  - [Utilisation](#utilisation)
    - [1. Collecte de données](#1-collecte-de-données)
    - [2. Prétraitement des données](#2-prétraitement-des-données)
    - [3. Entraînement du modèle](#3-entraînement-du-modèle)
    - [4. Reconnaissance en temps réel](#4-reconnaissance-en-temps-réel)
    - [5. Autres Scripts](#5-autres-scripts)
    - [6. Interface utilisateur Tkinter](#6-interface-utilisateur-tkinter)
    - [7. Interface utilisateur Flask](#7-interface-utilisateur-flask)
  - [Commandes Clavier](#commandes-clavier)
  - [Visualisation des Résultats](#visualisation-des-résultats)
  - [Mediapipe et les Landmarks](#mediapipe-et-les-landmarks)
    - [Mediapipe](#mediapipe)
    - [Landmarks](#landmarks)
    - [Fonctionnement dans le projet](#fonctionnement-dans-le-projet)
    - [Exemple de landmarks](#exemple-de-landmarks)
  - [Améliorations Futures](#améliorations-futures)
  - [Utiles](#utiles)
  - [Auteur](#auteur)


## Description
J'ai réalisé ce projet innovant dans le cadre du programme **M2 MIAGE parcours IA2 (Intelligence Artificielle Appliquée) 2024-2025** dans le module de "Computer Vision" avec **Monsieur Winter**.
Dans ce projet, j'utilise la vision par ordinateur et l'intelligence artificielle pour convertir des gestes de la langue des signes en texte et en parole. Il prend en charge l'alphabet (vers ce lien [Alphabets](images/alphabet.jpg)), les chiffres (1-9) (vers ce lien [Chiffres](images/Chiffres.png)) et des gestes spécifiques (vers ce lien [Quesques Gestes](images/Gestes.jpeg)). L'application comprend une interface utilisateur interactive et est conçue pour les besoins éducatifs et sociaux.

C'est un projet très complexe et c'est aussi difficile de trouver assez de données structurées pouvant nous servir d'entrainer et de valider notre modèle de réseau neurone.
Ainsi, pour pallier cela j'ai décider de collecter mes propres données avec la bliothèques `Mediapipe` avec les `Landmarks`. Ce qui est vachement plus intéressantes puisque pendant la collecte, j'ai pu la faire avec les deux mains, différentes positions et postures. Ce qui me permet d'avoir suffisamment de données diverses et variées au nombre de **2000** pour chaque **caratère** ou **élément** (**toutes les lettres d'alphabet**, **les chiffres de 1 à 9** ainsi que **17 gestes spécifiques**).

J'ai voulu déployé l'application complète sur Render ou Heroku mais lors du déploiement de l'application, plusieurs erreurs ont empêché le bon fonctionnement du backend.

1. **Temps d'exécution trop long (Worker Timeout)** : Cela peut être dû aux tâches lourdes suivantes :
    - Traitement vidéo avec OpenCV (lecture et traitement des frames en temps réel).
    - Utilisation de modèles TensorFlow/Keras pour effectuer des prédictions à chaque frame.

2. **Utilisation excessive des ressources** : Le traitement vidéo en temps réel avec OpenCV ou WebRTC, associé à un modèle d'apprentissage, consomme beaucoup de ressources CPU et mémoire. Les serveurs gratuits dont je dispose ont des ressources limitées, ce qui peut forcer l'arrêt des processus pour éviter une surcharge.

De ce fait, j'ai du déployer que le front finalement et j'ai mis deux options de lancements :
- Soit, on peut lancer l'application complète **Python Flash avec une interface intégrée** directement avec **```python app-local.py```**
- Soit on peut aller sur l'url du front déployé sur Render et lancer le backen **```python app.py```** d'un côté pour l'utiliser correctement.

---

- **Lien du dépôt Backend** : [Reconnaissance du langage des signes](https://github.com/dioprawane/sign_language_computer_vision_IA2)
- **Frontend** : [Lien vers le repo Git du frontend](https://github.com/dioprawane/front_sign_language_computer_vision_IA2)
- **Frontend déployé** : [Lien vers le front déployé sur Render](https://front-sign-language-computer-vision-ia2.onrender.com/)

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
│   ├── index.html             # Interface HTML de Flask du script app-local.py
│
├── utils/
│   ├── choix_model.md         # Choix du modèle
│
├── .gitignore                 # Fichiers à ignorer par Git
├── app.py                     # Application Flask (backend) du front déployé
├── app-local.py               # Application Flask avec son interface (pour local)
├── README.md                  # Documentation principale
├── requirements.txt           # Dépendances Python
└── venv/                      # Environnement virtuel Python
```

## Prérequis & Installation
### Prérequis
Avant de commencer, assurez-vous que votre environnement respecte les conditions suivantes :
- **Python 3.7 ou supérieur** installé sur votre machine.
- Une **caméra fonctionnelle** pour la reconnaissance en temps réel.
- Un accès à **pip**, le gestionnaire de paquets Python.
- (Optionnel) **Git** pour cloner le dépôt.

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
cd src
python collect_data.py
```
Ce script utilise la caméra pour capturer des données de landmarks correspondant aux gestes effectués. Les gestes sont définis dans la liste gestures du script. Chaque geste est enregistré sous forme de fichiers CSV dans le dossier `datasets/gesture_data/`. Ajoutez les gestes désirés dans la liste `gestures` pour personnaliser la collecte.

### 2. Prétraitement des données
Préparez les données collectées :
```bash
cd src
python pretreatment.py
```
Ce script lit les fichiers de landmarks collectés et les traite pour les diviser en ensembles d'entraînement et de test. Cela génère des fichiers prêts pour l'entraînement dans `datasets/processed_data/`.

### 3. Entraînement du modèle
Entraînez un modèle de reconnaissance des gestes :
```bash
cd src
python train_model.py
```
Ce script utilise les données prétraitées pour entraîner un réseau de neurones entièrement connecté capable de reconnaître les gestes.

* Le modèle entraîné est sauvegardé dans `models/gesture_model_V6.h5`.
* Les labels des classes sont sauvegardés dans `models/label_classes_V6.npy`.
* Ce script affiche également les métriques de performance et génère une matrice de confusion.

### 4. Reconnaissance en temps réel
Lancez la reconnaissance :
```bash
cd src
python recognize.py
```
Ce script utilise Mediapipe et le modèle entraîné pour détecter les gestes en temps réel à partir de la caméra. Les gestes détectés sont affichés sur la vidéo en temps réel avec OpenCV.

### 5. Autres Scripts
* ```config.py```
  * Configure les chemins des fichiers du modèle et des classes.
  * Définit les paramètres de Mediapipe pour la détection des mains.
  * Initialise les variables globales utilisées dans les autres scripts.
  * Configure la synthèse vocale avec pyttsx3.

* ```detection.py```
  * Gère la capture des frames de la caméra et le traitement des gestes avec Mediapipe et Keras.
  * Met à jour l'interface utilisateur en temps réel en utilisant Tkinter.
  * Intègre des fonctions pour ajouter des espaces ou des points automatiquement à partir des gestes détectés.
  
* ```gestures.py```
  * Fournit des fonctions utilitaires pour :
    * Réinitialiser la phrase et le mot courant.
    * Ajouter un espace ou un point.
    * Supprimer le dernier caractère.
    * Activer la synthèse vocale pour lire la phrase courante.
    * Basculer entre les modes pause et lecture.

* **`templates/index.html`** : Interface utilisateur web pour l'application Python Flask.

### 6. Interface utilisateur Tkinter
Lancez le script en local pour utiliser l'application en local :
```bash
cd src
python main.py
```
* Permet de reconnaître les gestes en temps réel en affichant le geste détecté directement sur le flux vidéo avec OpenCV.
* Implémente une interface utilisateur avec Tkinter pour afficher les gestes détectés, le mot courant, et la phrase construite.
* Permet une interaction avec la caméra pour la reconnaissance des gestes en temps réel.
* Ajoute des raccourcis clavier pour faciliter l'interaction.
* Utilise les autres scripts décrits précedemment pour fournir cette solution complète.

### 7. Interface utilisateur Flask
Lancez le serveur Flask pour utiliser l'application web avec une interface intégrée :
```bash
python app-local.py
```
Accédez à [http://127.0.0.1:5000](http://127.0.0.1:5000) dans le navigateur.

### 8. Soit avec l'interface utilisateur frontend et le back Python Flask
- Lancez le serveur Flask pour utiliser le frontend :
  ```bash
  python app.py
  ```
- Allez vers l'url du **frontend déployé** vers ce [lien](https://front-sign-language-computer-vision-ia2.onrender.com/)


## Commandes Clavier
- **`R`** : Réinitialiser
- **`Q`** : Quitter
- **`P`** : Parler
- **`S`** : Supprimer le dernier caractère
- **Espace** : Ajouter un espace
- **`.`** : Ajouter un point

## Visualisation des Résultats
L'application offre :
1. **Flux vidéo en temps réel** : Les gestes sont détectés et affichés en direct.
2. **Statut en texte** : Caractères, mots et phrases affichés dynamiquement.
3. **Synthèse vocale** : Lecture des phrases détectées.

## Mediapipe et les Landmarks

### Mediapipe
Mediapipe est une bibliothèque open-source développée par Google, qui fournit des solutions prêtes à l'emploi pour l'analyse de vidéos en temps réel. Elle est principalement utilisée dans les applications de vision par ordinateur, comme la détection et le suivi des mains, du visage ou du corps.

Dans ce projet, Mediapipe est utilisé pour :
- **Détecter les mains** dans une image capturée par la caméra.
- **Identifier les points clés (landmarks)** des mains, comme les articulations et les extrémités des doigts.

Mediapipe fonctionne en deux étapes :
1. **Détection de la main** : Mediapipe identifie la région de l'image contenant une ou plusieurs mains.
2. **Estimation des landmarks** : Une fois la main détectée, Mediapipe déduit les positions des points clés ou points de repère (landmarks) en fonction de la structure de la main.

### Landmarks
Les landmarks sont des points de repère spécifiques sur une image ou une vidéo, utilisés pour représenter des caractéristiques clés d'un objet détecté. Pour les mains, Mediapipe génère 21 landmarks, chacun correspondant à une articulation ou à une extrémité d'un doigt.

Chaque landmark est défini par trois coordonnées :
- **X** : Position horizontale du point dans l'image.
- **Y** : Position verticale du point dans l'image.
- **Z** : Profondeur du point, qui représente la distance par rapport à la caméra.

Ces landmarks permettent de :
- **Représenter la position et la structure de la main** dans l'espace 3D.
- **Suivre les mouvements des doigts et des mains**.
- **Extraire des données pour entraîner un modèle de reconnaissance des gestes.**

### Fonctionnement dans le projet
1. **Capture vidéo** :
   - La caméra capture un flux vidéo en temps réel.
   - Mediapipe détecte les mains dans chaque frame.

2. **Extraction des landmarks** :
   - Pour chaque main détectée, Mediapipe calcule les 21 landmarks.
   - Ces landmarks sont normalisés (entre 0 et 1) par rapport aux dimensions de l'image.

3. **Utilisation des landmarks** :
   - Les coordonnées des landmarks sont enregistrées dans des fichiers CSV pour la phase de collecte de données.
   - Pendant la reconnaissance, ces landmarks servent d'entrée au modèle de machine learning pour prédire le geste correspondant.

### Exemple de landmarks
Voici une visualisation des landmarks détectés sur une main :
![landmarks](images/landmarks.png)
Chaque point représente une articulation ou une extrémité des doigts :
- (0) : Poignet.
- (1)-(4) : Pouce.
- (5)-(8) : Index.
- (9)-(12) : Majeur.
- (13)-(16) : Annulaire.
- (17)-(20) : Auriculaire.

Ces landmarks sont utilisés pour capturer des gestes et les convertir en données exploitables pour la reconnaissance de la langue des signes.

## Améliorations Futures
- Ajout d'une base de données pour sauvegarder les sessions utilisateur.
- Détection multi-main et amélioration de la précision des gestes complexes.
- Ajout d'une API REST pour intégrer la reconnaissance dans d'autres applications.

## Utiles
![alphabet](images/alphabet.jpg)

![chiffres](images/Chiffres.png)

![gestes](images/Gestes.jpeg)

## Auteur
[Serigne Rawane Diop](https://github.com/dioprawane)

**Etudiant en Master 2 MIAGE parcours IA2**

**Apprenti chargé d'études statistiques et développeur en C# à l'Urssaf Caisse Nationale (Acoss)**

**Etudiant-entrepreneur**

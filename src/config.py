# src/config.py
import os
import time
import numpy as np
import pyttsx3
import mediapipe as mp
from tensorflow.keras.models import load_model

# =============================================================================
# 1) Déterminer le chemin ABSOLU du répertoire "config.py"
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 2) Construire les chemins d'accès au modèle et au fichier .npy
#    Ici, on suppose que "models/" est au même niveau que "src/", 
#    donc on fait BASE_DIR/.. pour remonter d'un cran, 
#    puis "models/gesture_model_V5.h5"
# =============================================================================
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "gesture_model_V5.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "..", "models", "label_classes_V5.npy")

# =============================================================================
# 3) Charger le modèle Keras et les classes
# =============================================================================
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

# =============================================================================
# 4) Mediapipe configuration
# =============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

# =============================================================================
# 5) Text-to-Speech setup
# =============================================================================
engine = pyttsx3.init()

# =============================================================================
# 6) Variables globales
# =============================================================================
current_alphabet = None
current_word = None
current_sentence = None
is_paused = None
word_buffer = ""
sentence = ""

# Pour gérer le délai de détection
last_detection_time = time.time()
RECOGNITION_DELAY = 3.0

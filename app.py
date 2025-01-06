from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
#CORS(app)  # Autoriser les requêtes cross-origin (pour un front séparé)

# ------------------ Chargement du modèle et config ------------------

MODEL_PATH = os.path.join("models", "gesture_model_V5.h5")
CLASSES_PATH = os.path.join("models", "label_classes_V5.npy")
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Variables globales
word_buffer = ""
sentence = ""
current_alphabet = "N/A"
last_detection_time = time.time()
RECOGNITION_DELAY = 3.0
is_paused = False

# ------------------ Text-to-speech (optionnel) ------------------
def speak_text(text):
    def tts_thread():
        try:
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except Exception as e:
            print(f"Error in TTS: {e}")

    threading.Thread(target=tts_thread, daemon=True).start()

# ------------------ Traitement du frame ------------------
def process_frame(frame_bgr):
    """
    - Convertit l'image BGR en RGB pour Mediapipe
    - Cherche les landmarks de la main
    - Fait la prédiction IA
    - Dessine les landmarks sur l'image
    - Renvoie (annotated_frame_bgr, current_alphabet, word_buffer, sentence)
    """
    global current_alphabet, word_buffer, sentence, last_detection_time, is_paused

    # Si pause, on ne détecte rien
    if is_paused:
        return frame_bgr

    # Conversion BGR -> RGB pour Mediapipe
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    current_time = time.time()

    # On vérifie s'il y a au moins une main détectée
    if results.multi_hand_landmarks and (current_time - last_detection_time >= RECOGNITION_DELAY):
        for hand_landmarks in results.multi_hand_landmarks:
            # Récupération des points normalisés (x, y, z)
            coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            try:
                prediction = model.predict([np.array(coords).flatten()[None, ...]])
                predicted_class = classes[np.argmax(prediction)].strip().lower()

                print(f"Detected alphabet: {predicted_class}")
                current_alphabet = predicted_class

                if current_alphabet in [" ", "espace"]:
                    sentence += word_buffer + " "
                    word_buffer = ""
                elif current_alphabet == "point":
                    word_buffer += "."
                else:
                    word_buffer += current_alphabet

                last_detection_time = current_time

            except Exception as e:
                print(f"Error in prediction: {e}")

            # Dessin des landmarks sur l'image (annotation)
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=2)
            )

    return frame_bgr

# ------------------ Routes ------------------

@app.route('/')
def index():
    return (
        "Bienvenue dans votre backend du langage des signes "
        "Utilise /webrtc_feed with POST frames (Base64)"
    )

@app.route('/webrtc_feed', methods=['POST'])
def webrtc_feed():
    """
    Reçoit périodiquement des images encodées en Base64 depuis le frontend (WebRTC).
    - Décode l'image
    - Fait la prédiction
    - Dessine les landmarks
    - Rencode l'image annotée en Base64 et la renvoie en JSON
    """
    global word_buffer, sentence, current_alphabet

    # Récupération du JSON
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data provided'}), 400

    frame_data = data['frame']
    # Retire le préfixe "data:image/...;base64,"
    if ',' in frame_data:
        frame_data = frame_data.split(',', 1)[1]

    try:
        # Décodage base64 -> bytes
        decoded = base64.b64decode(frame_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Traitement (détection + dessin landmarks)
        annotated_bgr = process_frame(frame_bgr)

        # Ré-encodage de l'image annotée en base64
        _, buffer_annotated = cv2.imencode('.jpg', annotated_bgr)
        annotated_b64 = base64.b64encode(buffer_annotated).decode('utf-8')
        annotated_frame_str = f"data:image/jpeg;base64,{annotated_b64}"

        # On renvoie aussi l'état courant
        return jsonify({
            'annotated_frame': annotated_frame_str,
            'current_alphabet': current_alphabet,
            'word_buffer': word_buffer,
            'sentence': sentence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_status', methods=['GET'])
def get_status():
    return jsonify({
        'current_alphabet': current_alphabet,
        'word_buffer': word_buffer,
        'sentence': sentence
    })

@app.route('/reset', methods=['POST'])
def reset():
    global word_buffer, sentence, current_alphabet
    word_buffer = ""
    sentence = ""
    current_alphabet = "N/A"
    return jsonify({'status': 'reset'})

@app.route('/remove_last', methods=['POST'])
def remove_last():
    global word_buffer
    if word_buffer:
        word_buffer = word_buffer[:-1]
    return jsonify({'word_buffer': word_buffer})

@app.route('/add_space', methods=['POST'])
def add_space():
    global word_buffer, sentence
    sentence += word_buffer + " "
    word_buffer = ""
    return jsonify({'sentence': sentence})

@app.route('/add_point', methods=['POST'])
def add_point():
    global word_buffer
    word_buffer += "."
    return jsonify({'word_buffer': word_buffer})

@app.route('/quit_and_show', methods=['POST'])
def quit_and_show():
    global sentence
    final_sentence = sentence
    return jsonify({'final_sentence': final_sentence})

@app.route('/speak', methods=['POST'])
def speak():
    speak_text(sentence)
    return jsonify({'status': 'speaking'})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({'status': 'paused' if is_paused else 'playing'})

# ------------------ Lancement ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
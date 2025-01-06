from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import threading
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pyttsx3
import time
import os
import base64

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Paths for the model and classes
MODEL_PATH = os.path.join("models", "gesture_model_V5.h5")
CLASSES_PATH = os.path.join("models", "label_classes_V5.npy")

# Load the model and classes
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

# Variables globales
word_buffer = ""
sentence = ""
current_alphabet = "N/A"
last_detection_time = time.time()
RECOGNITION_DELAY = 3.0
is_paused = False  # État de pause de la détection

# Ouvre la webcam via OpenCV (caméra index 0)
cap = cv2.VideoCapture(0)

def speak_text(text):
    """Prononcer le texte (TTS) via pyttsx3."""
    def tts_thread():
        try:
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except Exception as e:
            print(f"Error in TTS: {e}")

    threading.Thread(target=tts_thread, daemon=True).start()


def process_frame(frame):
    global word_buffer, sentence, current_alphabet
    global last_detection_time, is_paused

    if is_paused:
        return frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    if results.multi_hand_landmarks and (current_time - last_detection_time >= RECOGNITION_DELAY):
        for hand_landmarks in results.multi_hand_landmarks:
            # ... Logique de prédiction ...

    #return frame

            # Conversion en RGB pour Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_time = time.time()
            # Détection toutes les RECOGNITION_DELAY secondes
            if results.multi_hand_landmarks and (current_time - last_detection_time >= RECOGNITION_DELAY):
                for hand_landmarks in results.multi_hand_landmarks:
                    # Récupère les coordonnées 3D de la main
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                    try:
                        # Prédiction du modèle
                        prediction = model.predict(np.array([np.array(landmarks).flatten()]))
                        predicted_class = classes[np.argmax(prediction)]
                        current_alphabet = predicted_class.strip().lower()

                        print(f"Detected alphabet: {current_alphabet}")

                        if current_alphabet in [" ", "espace"]:
                            # Ajoute le mot courant à la phrase
                            sentence += word_buffer + " "
                            word_buffer = ""
                        elif current_alphabet == "point":
                            word_buffer += "."
                        else:
                            word_buffer += current_alphabet

                        # Mise à jour du temps de dernière détection
                        last_detection_time = current_time

                    except Exception as e:
                        print(f"Error in prediction: {e}")

                    # Dessin des landmarks sur l'image
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

    return frame


@app.route('/')
def index():
    """
    Page d'accueil : renvoie un template (index.html) 
    avec des variables Jinja si besoin.
    """
    return render_template(
        'index.html',
        current_alphabet=current_alphabet,
        word_buffer=word_buffer,
        sentence=sentence
    )


@app.route('/video_feed')
def video_feed():
    """
    Renvoie le flux vidéo MJPEG (OpenCV) au navigateur.
    """
    def generate_frames():
        global cap
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Effet miroir
            frame = cv2.flip(frame, 1)

            # Traitement du frame
            frame = process_frame(frame)

            # Encodage JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Envoi d'un chunk MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# --- Endpoints REST pour la logique (réinitialisation, etc.) ---

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
    status = "paused" if is_paused else "playing"
    return jsonify({'status': status})


# --- Lancement ---

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
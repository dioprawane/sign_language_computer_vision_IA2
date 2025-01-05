from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
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

# Flask setup and socketio setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Détermine si on est en local ou non
# Par exemple, lancez votre appli en local avec :  ENV_MODE=LOCAL python app.py
# Et en production : ENV_MODE=DEPLOYED python app.py
ENV_MODE = os.environ.get("ENV_MODE", "LOCAL")
IS_LOCAL = (ENV_MODE.upper() == "LOCAL")
print(f"ENV_MODE={ENV_MODE} | IS_LOCAL={IS_LOCAL}")

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

# Variables
word_buffer = ""
sentence = ""
current_alphabet = "N/A"
last_detection_time = time.time()
RECOGNITION_DELAY = 3.0
is_paused = False  # Variable pour gérer l'état de pause

# Si on est en local, on ouvre la caméra
if IS_LOCAL:
    cap = cv2.VideoCapture(0)
else:
    cap = None  # On n'utilise pas OpenCV si on est en mode déployé


def speak_text(text):
    """Fonction pour parler le texte donné via pyttsx3"""
    def tts_thread():
        try:
            local_engine = pyttsx3.init()  # Crée une nouvelle instance locale
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()  # S'assure que le moteur est correctement arrêté
        except Exception as e:
            print(f"Error in TTS: {e}")

    threading.Thread(target=tts_thread, daemon=True).start()


def process_frame(frame):
    """
    Fonction qui reçoit une image OpenCV (frame BGR),
    exécute Mediapipe et le modèle, et met à jour les
    variables globales : word_buffer, sentence, current_alphabet, etc.
    """
    global word_buffer, sentence, current_alphabet, last_detection_time, is_paused

    if is_paused:
        return frame  # On ne fait rien si c'est en pause

    # Conversion en RGB pour Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    if results.multi_hand_landmarks and (current_time - last_detection_time >= RECOGNITION_DELAY):
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks and predict
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            try:
                prediction = model.predict(np.array([np.array(landmarks).flatten()]))
                predicted_class = classes[np.argmax(prediction)]

                print(f"Raw predictions: {prediction}")
                print(f"Predicted class: {predicted_class}")

                # Update global variables
                current_alphabet = predicted_class.strip().lower()
                print(f"Detected alphabet: {current_alphabet}")

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

            # Dessiner les landmarks sur l'image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame


@app.route('/')
def index():
    return render_template('index.html',
                           current_alphabet=current_alphabet,
                           word_buffer=word_buffer,
                           sentence=sentence)


@app.route('/video_feed')
def video_feed():
    """
    Si on est en local, on renvoie le flux vidéo via OpenCV.
    Sinon, on renvoie un flux ou un message vide, car c’est WebRTC qui gère la vidéo.
    """
    if not IS_LOCAL:
        # En mode déployé, on n'utilise pas OpenCV, donc on peut retourner un vide ou un message
        return jsonify({'info': 'Not in local mode, use WebRTC feed instead.'})

    def generate_frames():
        global cap
        while True:
            success, frame = cap.read()
            if not success:
                break

            # On retourne horizontalement l'image pour un rendu miroir
            frame = cv2.flip(frame, 1)

            # Traitement du frame
            frame = process_frame(frame)

            # Encodage en JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webrtc_feed', methods=['POST'])
def webrtc_feed():
    """
    Endpoint appelé en mode déployé pour recevoir périodiquement
    des images encodées en Base64 depuis le navigateur (WebRTC).
    """
    if IS_LOCAL:
        # En mode local, on n'utilise pas ce endpoint
        return jsonify({'info': 'Local mode uses OpenCV capture, not WebRTC.'}), 400

    global word_buffer, sentence, current_alphabet, last_detection_time, is_paused

    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data provided'}), 400

    frame_data = data['frame']  # Ex: "data:image/jpeg;base64,...."
    try:
        # On retire la partie "data:image/...;base64," s’il y en a une
        if ',' in frame_data:
            frame_data = frame_data.split(',', 1)[1]

        decoded = base64.b64decode(frame_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Traitement du frame avec la même fonction que pour OpenCV
        process_frame(frame)

        # On peut renvoyer l’état actuel
        return jsonify({
            'current_alphabet': current_alphabet,
            'word_buffer': word_buffer,
            'sentence': sentence
        })
    except Exception as e:
        return jsonify({'error': f'Exception while processing frame: {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset():
    global word_buffer, sentence, current_alphabet
    word_buffer = ""
    sentence = ""
    current_alphabet = "N/A"
    return jsonify({'status': 'reset'})


@app.route('/speak', methods=['POST'])
def speak():
    global sentence
    speak_text(sentence)
    return jsonify({'status': 'speaking'})


@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    status = "paused" if is_paused else "playing"
    return jsonify({'status': status})


@app.route('/quit_and_show', methods=['POST'])
def quit_and_show():
    global sentence
    final_sentence = sentence
    return jsonify({'final_sentence': final_sentence})


@app.route('/remove_last', methods=['POST'])
def remove_last():
    global word_buffer
    word_buffer = word_buffer[:-1] if word_buffer else ""
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


@app.route('/get_status', methods=['GET'])
def get_status():
    global current_alphabet, word_buffer, sentence
    return jsonify({
        'current_alphabet': current_alphabet,
        'word_buffer': word_buffer,
        'sentence': sentence
    })


if __name__ == '__main__':
    # En local, vous pouvez appeler directement 'app.run(debug=True)' ou utiliser socketio.run.
    # Note : évitez de lancer socketio.run() deux fois. Choisissez-en un seul.
    # Ci-dessous, on peut faire :
    #   python app.py -> lance le serveur Flask en local
    # Pour un usage plus avancé, vous pouvez aussi faire :
    #   socketio.run(app, debug=True)

    print(f"ENV_MODE={ENV_MODE} | IS_LOCAL={IS_LOCAL}")
    app.run(debug=True, host='0.0.0.0', port=5000)
    # socketio.run(app, debug=True)
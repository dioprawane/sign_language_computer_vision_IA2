from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pyttsx3
import time
import os

# Flask setup
app = Flask(__name__)

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

# Capture video from webcam
cap = cv2.VideoCapture(0)

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

@app.route('/')
def index():
    return render_template('index.html', current_alphabet=current_alphabet, word_buffer=word_buffer, sentence=sentence)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global word_buffer, sentence, current_alphabet, last_detection_time, is_paused

        while True:
            if is_paused:
                time.sleep(0.1)  # Réduire l'utilisation CPU en mode pause
                continue

            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
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

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    app.run(debug=True)
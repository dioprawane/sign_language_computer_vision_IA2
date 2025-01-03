# src/detection.py
import time
import cv2
from PIL import Image, ImageTk
import numpy as np

# On importe mp_hands et mp_drawing depuis config.py (où vous les avez définis)
from config import model, classes, mp_hands, mp_drawing
import config as config
from gestures import add_space_manually, add_point_manually

def process_frame(root, video_label):
    """
    Capture webcam, traitement MediaPipe/Keras,
    mise à jour de l'interface Tkinter.
    """
    cap = cv2.VideoCapture(0)

    # Configuration de la webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # On configure MediaPipe Hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

    while cap.isOpened():
        # Si en pause, on ne lit pas les frames
        if config.is_paused.get() == "True":
            root.update_idletasks()
            root.update()
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Miroir horizontal
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reconnaissance des gestes
        results = hands.process(rgb_frame)

        current_time = time.time()
        # Respect du délai minimal de détection
        if results.multi_hand_landmarks and (
            current_time - config.last_detection_time >= config.RECOGNITION_DELAY
        ):
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                # Préparation pour Keras
                prediction = model.predict(np.array([np.array(landmarks).flatten()]))

                gesture = classes[np.argmax(prediction)]
                config.current_alphabet.set(gesture)

                gesture_clean = gesture.strip().lower()
                if gesture_clean in [" ", "espace"]:
                    add_space_manually()
                elif gesture_clean == "point":
                    add_point_manually()
                else:
                    config.word_buffer += gesture
                    config.current_word.set(config.word_buffer)

                # Dessiner les landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            config.last_detection_time = current_time

        # Conversion pour Tkinter
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Si besoin, 
        # ou plus simplement :
        # img_bgr = frame 
        # selon la logique que vous voulez pour l'affichage
        img = ImageTk.PhotoImage(image=Image.fromarray(img_bgr))
        video_label.imgtk = img
        video_label.configure(image=img)

        root.update_idletasks()
        root.update()

    cap.release()
    cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle et les classes
#model = load_model("../models/gesture_model_V5.h5")
#classes = np.load("../models/label_classes_V5.npy")
model = load_model("C:/Users/ac75008612/perso/langage_des_signes/models/gesture_model_V5.h5")
classes = np.load("C:/Users/ac75008612/perso/langage_des_signes/models/label_classes_V5.npy")

# Configuration Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                prediction = model.predict(np.array([np.array(landmarks).flatten()]))
                gesture = classes[np.argmax(prediction)]
                cv2.putText(frame, f"Gesture : {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def main():
    print("Starting Gesture Recognition...")
    recognize_gesture()

if __name__ == "__main__":
    recognize_gesture()
    main()
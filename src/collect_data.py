import cv2
import mediapipe as mp
import os
import shutil

# Configuration Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Variable globale pour gérer l'arrêt total
stop_all = False

def collect_data(label, output_dir="../datasets/gesture_data", num_samples=2000):
    """
    Fonction pour capturer les données des landmarks pour un geste ou une lettre spécifique.

    Args:
        label (str): Le nom du geste ou de la lettre.
        output_dir (str): Le dossier où sauvegarder les données.
        num_samples (int): Nombre d'échantillons à capturer.
    """
    global stop_all
    dir_path = f"{output_dir}/{label}"
    os.makedirs(dir_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Collecte data pour le geste : {label}")
    print("Appuyez sur 'q' pour quitter la capture de tous les gestes.")
    print("Appuyez sur 'r' pour recommencer la collecte des données.")

    while count < num_samples:
        if stop_all:  # Vérifie si l'arrêt total a été demandé
            break

        ret, frame = cap.read()
        if not ret:
            break
        
        # Retourne et convertit le frame en RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Dessiner les landmarks et collecter les données
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extraire les coordonnées des landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Sauvegarde des landmarks dans un fichier CSV
                save_path = f"{dir_path}/{count}.csv"
                with open(save_path, 'w') as f:
                    f.write(','.join(map(str, landmarks)) + '\n')
                count += 1
                
        # Redimensionner la taille du frame pour une meilleure visibilité
        frame = cv2.resize(frame, (800, 600))  # Taille personnalisée (largeur=1280, hauteur=720)

        # Afficher le frame avec les landmarks
        cv2.putText(frame, f"Geste : {label} | Echantillon : {count}/{num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"[Appuyez sur 'r' pour recommencer | 'q' pour tout quitter]", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Collecte de donnees", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quitter la collecte pour tous les gestes
        if key == ord('q'):
            stop_all = True
            break

        # Recommencer la collecte
        if key == ord('r'):
            print(f"Recommencer la collecte pour : {label}")
            shutil.rmtree(dir_path)  # Supprimer le dossier existant
            os.makedirs(dir_path, exist_ok=True)  # Le recréer
            count = 0  # Réinitialiser le compteur

    cap.release()
    cv2.destroyAllWindows()
    if stop_all:
        print("Arrêt de la collecte pour tous les gestes.")
    else:
        print(f"Collecte terminée pour : {label}")

if __name__ == "__main__":
    # Liste des gestes à capturer
    gestures = [
        # Lettres de l'alphabet
        *[chr(i) for i in range(65, 91)],  # A-Z
        
        # Chiffres
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        
        # Gestes spécifiques : 
        "appeler", "ne_pas_aimer", "aimer", "attraper", "ok", "paume", "pierre", "point", 
        "stop", "doigt_d_honneur", "pistolet", "prière", "temps_mort", "prendre_photo",
        "coeur", "deux_coeurs", "espace"
    ]

    # Nombre d'échantillons par catégorie
    samples_per_gesture = 2000

    # Collecte des données pour chaque geste
    for gesture in gestures:
        collect_data(label=gesture, num_samples=samples_per_gesture)
        if stop_all:  # Si l'arrêt global est demandé, sortir de la boucle
            break

    print("Programme terminé.")
import os
import numpy as np

def reshape_landmarks_to_pseudo_images(input_dir, output_dir, num_landmarks=21, channels=3):
    """
    Réorganise les fichiers CSV contenant les landmarks en pseudo-images (matrices).

    Args:
        input_dir (str): Répertoire contenant les fichiers CSV classés par catégories (chiffres, lettres, gestes).
        output_dir (str): Répertoire où sauvegarder les pseudo-images sous forme de fichiers .npy.
        num_landmarks (int): Nombre de points de repère (landmarks) par exemple.
        channels (int): Nombre de canaux (3 pour x, y, z par défaut).

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir les catégories (chiffres, lettres, gestes)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        print(f"Traitement de la catégorie : {category}")
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        # Parcourir les fichiers de cette catégorie
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if file_name.endswith(".csv"):
                # Charger les données depuis le fichier CSV
                data = np.loadtxt(file_path, delimiter=',')

                # Vérifier que la taille correspond bien
                if len(data) != num_landmarks * channels:
                    print(f"Fichier {file_name} ignoré (dimensions incorrectes).")
                    continue

                # Reshape en pseudo-image (h=num_landmarks, w=channels)
                pseudo_image = data.reshape((num_landmarks, channels))

                # Sauvegarder la pseudo-image dans un fichier .npy
                output_file = os.path.join(category_output_dir, file_name.replace(".csv", ".npy"))
                np.save(output_file, pseudo_image)

        print(f"Catégorie {category} traitée et sauvegardée dans {category_output_dir}")

if __name__ == "__main__":
    # Répertoires des données
    input_dir = "../datasets/gesture_data"  # Répertoire contenant les fichiers CSV classés par catégories
    output_dir = "../datasets/pseudo_images"  # Répertoire où sauvegarder les fichiers transformés

    # Paramètres des landmarks
    num_landmarks = 21  # Par défaut pour MediaPipe
    channels = 3  # x, y, z

    # Appeler la fonction pour transformer les fichiers CSV
    reshape_landmarks_to_pseudo_images(input_dir, output_dir, num_landmarks=num_landmarks, channels=channels)
    print("Réorganisation des données terminée.")
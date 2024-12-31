import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_by_category(data_dir, test_size=0.2, mix_ratio=0.1):
    """
    Charge les données et les classe en chiffres, lettres, et gestes.
    Mélange les données de test avec des exemples incorrects de la même catégorie.

    Args:
        data_dir (str): Répertoire contenant les données organisées en sous-dossiers.
        test_size (float): Proportion des données pour les tests.
        mix_ratio (float): Proportion de faux exemples dans les données de test.

    Returns:
        dict: Ensembles d'entraînement et de test pour chaque catégorie.
    """
    categories = ["numbers", "letters", "gestures"]
    datasets = {}

    for category in categories:
        print(f"\nDébut du traitement de la catégorie : {category}")

        # Définir le chemin du sous-dossier pour la catégorie
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"Le répertoire {category} n'existe pas.")
            continue

        features = []
        labels_array = []

        # Parcourir chaque sous-dossier dans la catégorie
        for label in os.listdir(category_dir):
            label_dir = os.path.join(category_dir, label)
            if not os.path.isdir(label_dir):  # Vérifier si c'est un répertoire
                continue

            print(f"  Traitement du label : {label}")
            for file in os.listdir(label_dir):
                filepath = os.path.join(label_dir, file)
                if file.endswith(".csv"):  # S'assurer que le fichier est un CSV
                    try:
                        data = pd.read_csv(filepath, header=None).values.flatten()
                        features.append(data)
                        labels_array.append(label)
                        print(f"    Fichier traité : {file}")
                    except Exception as e:
                        print(f"    Erreur lors de la lecture du fichier {file}: {e}")

        # Convertir en tableaux NumPy
        features = np.array(features)
        labels_array = np.array(labels_array)

        # Si aucune donnée n'a été trouvée pour cette catégorie, passer
        if len(features) == 0 or len(labels_array) == 0:
            print(f"Aucune donnée trouvée pour la catégorie : {category}")
            continue

        # Séparer les ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(features, labels_array, test_size=test_size, random_state=42)

        # Mélanger les données de test avec des exemples d'autres labels de la même catégorie
        num_to_mix = int(len(X_test) * mix_ratio)
        indices_to_mix = np.random.choice(len(X_test), size=num_to_mix, replace=False)

        for idx in indices_to_mix:
            current_label = y_test[idx]
            other_labels = [label for label in np.unique(labels_array) if label != current_label]
            if len(other_labels) == 0:
                continue  # Si pas d'autres labels dans cette catégorie, ignorer

            mixed_label = np.random.choice(other_labels)
            mixed_index = np.random.choice(np.where(labels_array == mixed_label)[0])

            # Remplacer les données avec les faux exemples
            X_test[idx] = features[mixed_index]
            y_test[idx] = labels_array[mixed_index]

        datasets[category] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        print(f"Fin du traitement de la catégorie : {category}")

    return datasets


if __name__ == "__main__":
    # Répertoire des données collectées
    data_dir = "../datasets/gesture_data"

    # Charger et mélanger les données
    print("Début du prétraitement des données...")
    datasets = load_data_by_category(data_dir)

    # Sauvegarder les données prétraitées
    output_dir = "../datasets/processed_data"
    os.makedirs(output_dir, exist_ok=True)

    for category, data in datasets.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        np.save(os.path.join(category_dir, "features_train.npy"), data["X_train"])
        np.save(os.path.join(category_dir, "features_test.npy"), data["X_test"])
        np.save(os.path.join(category_dir, "labels_train.npy"), data["y_train"])
        np.save(os.path.join(category_dir, "labels_test.npy"), data["y_test"])

    print("Prétraitement terminé avec mélange par catégorie.")
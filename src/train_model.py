import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def build_model(input_shape, num_classes):
    """
    Construit un modèle séquentiel pour la classification des gestes, lettres et chiffres.

    Args:
        input_shape (int): Taille de l'entrée.
        num_classes (int): Nombre de classes de sortie.

    Returns:
        Model: Modèle compilé Keras.
    """
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(96, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Charger les données combinées
    data_dir = "../datasets/processed_data"
    categories = ["numbers", "letters", "gestures"]

    features_list, labels_list = [], []

    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"Le répertoire {category} n'existe pas. Ignoré.")
            continue

        # Charger les données de la catégorie
        X_train = np.load(os.path.join(category_dir, "features_train.npy"))
        X_test = np.load(os.path.join(category_dir, "features_test.npy"))
        y_train = np.load(os.path.join(category_dir, "labels_train.npy"))
        y_test = np.load(os.path.join(category_dir, "labels_test.npy"))

        # Combiner les ensembles train et test pour un entraînement global
        features_list.append(np.vstack((X_train, X_test)))
        labels_list.append(np.hstack((y_train, y_test)))

    # Combiner toutes les catégories
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)

    # Encoder les labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    np.save("../models/label_classes_V5.npy", le.classes_)

    # Diviser les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Construire et entraîner le modèle
    model = build_model(input_shape=X_train.shape[1], num_classes=len(le.classes_))
    history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

    # Sauvegarder le modèle
    model.save("../models/gesture_model_V5.h5")
    print("Modèle sauvegardé sous ../models/gesture_model_V5.h5")

    # Visualiser les performances
    fig = plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Précision')
    plt.xlabel('Époque')
    plt.legend([f'Apprentissage: {round(100 * history.history["accuracy"][-1], 1)}%', 
                f'Test: {round(100 * history.history["val_accuracy"][-1], 1)}%'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pertes du modèle')
    plt.ylabel('Pertes')
    plt.xlabel('Époque')
    plt.legend([f'Apprentissage: {round(history.history["loss"][-1], 3)}', 
                f'Test: {round(history.history["val_loss"][-1], 3)}'], loc='lower right')

    plt.tight_layout()
    plt.show()

    # Prédictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Identification des exemples mal classés
    incorrect = np.where(y_pred != y_test)[0]

    # Afficher quelques exemples mal classés (en utilisant des vecteurs de caractéristiques)
    num_images = min(len(incorrect), 10)  # Afficher au maximum 10 exemples
    plt.figure(figsize=(10, 5))

    for i, incorrect_idx in enumerate(incorrect[:num_images]):
        plt.subplot(2, 5, i + 1)
        plt.bar(range(len(X_test[incorrect_idx])), X_test[incorrect_idx])  # Affiche sous forme d'histogramme
        plt.title(f"Prédit: {le.inverse_transform([y_pred[incorrect_idx]])[0]}\nVrai: {le.inverse_transform([y_test[incorrect_idx]])[0]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :")
    print(cm)

    # Affichage de la matrice de confusion
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matrice de confusion")
    plt.ylabel("Vraies classes")
    plt.xlabel("Classes prédites")
    plt.show()

    # Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
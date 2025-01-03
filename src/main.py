# main.py
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
import threading

# Imports internes
import config
from gestures import (
    speak_text,
    reset_sentence,
    remove_last_letter,
    add_space_manually,
    add_point_manually,
    quit_and_show_sentence,
    toggle_pause,
)
from detection import process_frame

def on_space_key(event):
    add_space_manually()

def on_r_key(event):
    remove_last_letter()

def on_q_key(event):
    quit_and_show_sentence(root)

def on_point_key(event):
    add_point_manually()


# ===================== Lancement principal =====================
if __name__ == "__main__":
    # -- On définit la fenêtre principale
    root = tk.Tk()
    root.title("Conversion de la langue des signes en texte et en parole")
    root.geometry("1300x650")
    root.configure(bg="#2c2f33")
    root.resizable(False, False)

    # -- Initialiser les variables Tkinter (dans config)
    config.current_alphabet = StringVar(root, value="N/A")
    config.current_word = StringVar(root, value="N/A")
    config.current_sentence = StringVar(root, value="N/A")
    config.is_paused = StringVar(root, value="False")

    # ================== Layout de l'interface ==================
    video_frame = Frame(root, bg="#2c2f33", bd=5, relief="solid", width=500, height=400)
    video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
    video_frame.grid_propagate(False)

    video_label = tk.Label(video_frame)
    video_label.pack(expand=True)

    content_frame = Frame(root, bg="#2c2f33")
    content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))

    Label(content_frame, text="Caractère courant :", font=("Arial", 20), fg="#ffffff", bg="#2c2f33")\
        .pack(anchor="w", pady=(0, 10))
    Label(content_frame, textvariable=config.current_alphabet, font=("Arial", 24, "bold"), fg="#1abc9c", bg="#2c2f33")\
        .pack(anchor="center")

    Label(content_frame, text="Mot en cours :", font=("Arial", 20), fg="#ffffff", bg="#2c2f33")\
        .pack(anchor="w", pady=(20, 10))
    Label(content_frame, textvariable=config.current_word, font=("Arial", 16), fg="#f39c12", bg="#2c2f33",
          wraplength=500, justify="left")\
        .pack(anchor="center")

    Label(content_frame, text="Phrase en cours :", font=("Arial", 20), fg="#ffffff", bg="#2c2f33")\
        .pack(anchor="w", pady=(20, 10))
    Label(content_frame, textvariable=config.current_sentence, font=("Arial", 16), fg="#9b59b6", bg="#2c2f33",
          wraplength=500, justify="left")\
        .pack(anchor="center")

    button_frame = Frame(root, bg="#2c2f33")
    button_frame.grid(row=3, column=1, pady=(10, 20), padx=(10, 20), sticky="n")

    reset_button = Button(button_frame, text="Réinitialiser", font=("Arial", 16),
                          command=reset_sentence, bg="#e74c3c", fg="#ffffff", relief="flat", height=2, width=14)
    reset_button.grid(row=0, column=0, padx=10)

    pause_button = Button(button_frame, text="Pause", font=("Arial", 16),
                          command=lambda: toggle_pause(pause_button),
                          bg="#3498db", fg="#ffffff", relief="flat", height=2, width=14)
    pause_button.grid(row=0, column=1, padx=10)

    speak_button = Button(button_frame, text="Parler", font=("Arial", 16),
                          command=lambda: speak_text(config.current_sentence.get()),
                          bg="#27ae60", fg="#ffffff", relief="flat", height=2, width=14)
    speak_button.grid(row=0, column=2, padx=10)

    remove_letter_button = Button(button_frame, text="Supprimer Dernier", font=("Arial", 16),
                                  command=remove_last_letter, bg="#d35400", fg="#ffffff",
                                  relief="flat", height=2, width=14)
    remove_letter_button.grid(row=1, column=0, padx=10, pady=5)

    add_space_button = Button(button_frame, text="Ajouter Espace", font=("Arial", 16),
                              command=add_space_manually, bg="#8e44ad", fg="#ffffff",
                              relief="flat", height=2, width=14)
    add_space_button.grid(row=1, column=1, padx=10, pady=5)

    add_point_button = Button(button_frame, text="Ajout Point", font=("Arial", 16),
                              command=add_point_manually, bg="#2ecc71", fg="#ffffff",
                              relief="flat", height=2, width=14)
    add_point_button.grid(row=1, column=2, padx=10, pady=5)

    quit_button = Button(button_frame, text="Quitter", font=("Arial", 16),
                         command=lambda: quit_and_show_sentence(root), bg="#c0392b", fg="#ffffff",
                         relief="flat", height=2, width=14)
    quit_button.grid(row=2, column=1, padx=10, pady=5)

    # ================== Liaison des touches du clavier ==================
    root.bind("<space>", on_space_key)
    root.bind("r", on_r_key)
    root.bind("q", on_q_key)
    root.bind("<period>", on_point_key)

    # ================== Lancer le traitement vidéo ==================
    threading.Thread(target=process_frame, args=(root, video_label), daemon=True).start()

    # ================== Lancer l'interface ==================
    root.mainloop()
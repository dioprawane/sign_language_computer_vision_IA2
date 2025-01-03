# gestures.py
import threading
import config

# Fonctions utilitaires

def speak_text(text):
    """
    Lance la synthèse vocale dans un thread séparé.
    """
    def tts_thread():
        config.engine.say(text)
        config.engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

def reset_sentence():
    """
    Réinitialiser le mot et la phrase.
    """
    config.word_buffer = ""
    config.sentence = ""
    config.current_word.set("N/A")
    config.current_sentence.set("N/A")
    config.current_alphabet.set("N/A")

def remove_last_letter():
    """
    Enlever la dernière lettre du mot en cours.
    """
    if len(config.word_buffer) > 0:
        config.word_buffer = config.word_buffer[:-1]
    config.current_word.set(config.word_buffer if config.word_buffer else "N/A")

def add_space_manually():
    """
    Ajouter un espace (valider le mot).
    """
    if config.word_buffer.strip():
        config.sentence += config.word_buffer + " "
        config.current_sentence.set(config.sentence.strip())
    config.word_buffer = ""
    config.current_word.set("N/A")

def add_point_manually():
    """
    Ajouter un point (.) au mot en cours.
    """
    config.word_buffer += "."
    config.current_word.set(config.word_buffer)

def quit_and_show_sentence(root):
    """
    Quitter et afficher la phrase finale en console.
    """
    print("Phrase finale :", config.current_sentence.get())
    root.quit()

def toggle_pause(pause_button):
    """
    Basculer entre pause et lecture.
    """
    if config.is_paused.get() == "False":
        config.is_paused.set("True")
        pause_button.config(text="Play")
    else:
        config.is_paused.set("False")
        pause_button.config(text="Pause")
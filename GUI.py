
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
from tensorflow import keras
import pickle

# ========== Load Model & Mappings ==========
with open(r"D:\f\Level 4\Second term\NLP\project\project\word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open(r"D:\f\Level 4\Second term\NLP\project\project\idx2tag.pkl", "rb") as f:
    idx2tag = pickle.load(f)

max_len = 50
model = keras.models.load_model(r"D:\f\Level 4\Second term\NLP\project\project\NER_model.keras")

# ========== Prediction Function ==========
def predict_sentence(sentence):
    tokens = sentence.split()
    input_ids = [word2idx.get(w, word2idx.get("UNK", 1)) for w in tokens]
    padded_input = input_ids + [word2idx["PAD"]] * (max_len - len(input_ids))
    padded_input = np.array(padded_input).reshape(1, max_len)
    pred = model.predict(padded_input)
    pred = np.argmax(pred, axis=-1)[0]
    tags = [idx2tag[idx] for idx in pred[:len(tokens)]]
    return list(zip(tokens, tags))

# ========== GUI Logic ==========
def on_predict():
    input_text = text_input.get("1.0", tk.END).strip()
    result = predict_sentence(input_text)

    text_output.config(state="normal")
    text_output.delete("1.0", tk.END)

    # Show (word -> tag) pairs
    pairs_text = "\n".join([f"{word} -> {tag}" for word, tag in result])
    text_output.insert(tk.END, "=== Word -> Tag Pairs ===\n", "header")
    text_output.insert(tk.END, pairs_text + "\n\n")

    # Prepare for highlighted text
    for tag in text_output.tag_names():
        if tag not in ["header"]:
            text_output.tag_delete(tag)

    tag_colors = {
        "O": None,
        "B-PERS": "#ffcccc",   # Red
        "I-PERS": "#ffcccc",
        "B-LOC": "#ffff99",   # Yellow
        "I-LOC": "#ffff99",
        "B-ORG": "#ccffcc",   # Green
        "I-ORG": "#ccffcc",
        "B-MISC": "#e0ccff",  # Purple
        "I-MISC": "#e0ccff",
    }


    for word, tag in result:
        start_index = text_output.index(tk.INSERT)
        text_output.insert(tk.END, word + " ")

        if tag in tag_colors and tag_colors[tag]:
            end_index = text_output.index(tk.INSERT)
            text_output.tag_add(tag, start_index, end_index)
            text_output.tag_config(tag, background=tag_colors[tag])

    text_output.tag_config("header", font=("Arial", 10, "bold"))

    text_output.config(state="disabled")

# Right-click paste support
def show_context_menu(event):
    context_menu.tk_popup(event.x_root, event.y_root)

def paste_text():
    text_input.event_generate("<<Paste>>")

# ========== GUI ==========
root = tk.Tk()
root.title("Arabic NER")

tk.Label(root, text="Enter Arabic Sentence:", anchor="e", font=("Arial", 10)).pack()

text_input = scrolledtext.ScrolledText(root, height=5, width=80, wrap=tk.WORD, font=("Arial", 12), undo=True)
text_input.pack(padx=10, pady=5)
text_input.bind("<Button-3>", show_context_menu)

context_menu = tk.Menu(root, tearoff=0)
context_menu.add_command(label="Paste", command=paste_text)

tk.Button(root, text="Predict", command=on_predict).pack(pady=5)

tk.Label(root, text="NER Output:", font=("Arial", 10)).pack()
text_output = scrolledtext.ScrolledText(root, height=15, width=80, wrap=tk.WORD, font=("Arial", 12), state="disabled")
text_output.pack(padx=10, pady=5)

root.mainloop()
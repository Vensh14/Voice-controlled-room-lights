import os
import queue
import threading
import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("voice_switch_fixed.keras")

# Labels
WORDS = ["on", "off", "one", "two", "three", "cat", "dog", "wow", "_background_noise_"]

# Parameters
SR = 16000
DURATION = 1  # seconds per audio chunk
MFCC_FEATURES = 40
MAX_LEN = 32
CONF_THRESHOLD = 0.90  # confidence threshold

# Queue for audio stream
q = queue.Queue()


# -----------------------------
# Preprocess audio chunk
# -----------------------------
def preprocess(signal, sr=SR):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc[..., np.newaxis][np.newaxis, ...]


def is_silence(signal, threshold=0.001):
    energy = np.mean(signal**2)
    return energy < threshold


# -----------------------------
# GUI Class
# -----------------------------
class RoomLightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Room Lights")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        # Light state
        self.lights = {1: False, 2: False, 3: False}

        # Load images
        self.images_on = {}
        self.images_off = {}
        for i in range(1, 4):
            self.images_on[i] = Image.open(f"room{i}_on.jpg")
            self.images_off[i] = Image.open(f"room{i}_off.jpg")

        # --- Welcome Label ---
        self.label = tk.Label(
            root,
            text="Welcome to Home Automation System\nSay 'ON' to start",
            font=("Arial", 26, "bold"),
            bg="#2c3e50",
            fg="white",
            justify="center",
        )
        self.label.pack(pady=40)

        # --- Feedback label ---
        self.feedback_label = tk.Label(
            root,
            text="",
            font=("Arial", 18),
            bg="#2c3e50",
            fg="red",
        )
        self.feedback_label.pack(side="bottom", pady=10)

        # --- Room Frame (hidden initially) ---
        self.room_frame = tk.Frame(root, bg="#34495e")
        self.room_labels = {}
        for i in range(1, 4):
            lbl = tk.Label(self.room_frame, bg="#34495e")
            lbl.grid(row=0, column=i - 1, padx=20, pady=20, sticky="nsew")
            self.room_labels[i] = lbl
        for i in range(3):
            self.room_frame.grid_columnconfigure(i, weight=1)

        # Start auto-resize loop
        self.root.after(200, self.auto_resize)

    def auto_resize(self):
        if self.room_frame.winfo_ismapped():
            self.resize_images()
        self.root.after(200, self.auto_resize)

    def resize_images(self):
        frame_width = self.room_frame.winfo_width() or self.root.winfo_width()
        frame_height = self.room_frame.winfo_height() or self.root.winfo_height()
        img_size = min(frame_width // 3 - 40, frame_height - 150)
        for i in range(1, 4):
            img = self.images_on[i] if self.lights[i] else self.images_off[i]
            resized = img.resize((img_size, img_size))
            tk_img = ImageTk.PhotoImage(resized)
            self.room_labels[i].config(image=tk_img)
            self.room_labels[i].image = tk_img

    def process_command(self, command):
        """Update GUI based on recognized command"""
        if command == "on":
            self.label.config(
                text="Say room number to switch on light\nSay 'OFF' to go back"
            )
            self.feedback_label.config(text="")
            self.room_frame.pack(pady=20, fill="both", expand=True)
            self.show_all_off()
        elif command == "off":
            self.label.config(
                text="Welcome to Home Automation System\nSay 'ON' to start"
            )
            self.feedback_label.config(text="")
            self.room_frame.pack_forget()
        elif command in ["one", "two", "three"] and self.room_frame.winfo_ismapped():
            room = ["one", "two", "three"].index(command) + 1
            self.lights[room] = not self.lights[room]
            self.resize_images()
            self.feedback_label.config(text="")
        else:
            self.feedback_label.config(text="❌ Try again!" if command else "")

    def show_all_off(self):
        for i in range(1, 4):
            self.lights[i] = False
        self.resize_images()


# -----------------------------
# Listening thread
# -----------------------------
def listen_and_predict(app):
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    with sd.InputStream(
        callback=audio_callback, channels=1, samplerate=SR, blocksize=int(SR * DURATION)
    ):
        print("🎤 Listening... (Ctrl+C to stop)")
        while True:
            audio_chunk = q.get().flatten()

            if is_silence(audio_chunk):
                continue

            features = preprocess(audio_chunk)
            pred = model.predict(features, verbose=0)
            idx = np.argmax(pred)
            confidence = np.max(pred)

            if confidence >= CONF_THRESHOLD:
                detected_word = WORDS[idx]
                if detected_word not in ["_background_noise_"]:
                    print(f"✅ Detected: {detected_word} (conf={confidence:.2f})")
                    app.root.after(0, app.process_command, detected_word)
            else:
                app.root.after(0, app.process_command, "")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RoomLightApp(root)

    # Run listening loop in background thread
    threading.Thread(target=listen_and_predict, args=(app,), daemon=True).start()

    root.mainloop()

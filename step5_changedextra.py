import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
import queue

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("voice_switch_fixed.keras")

# Labels
WORDS = ["on", "off", "one", "two", "three","cat","dog","wow","_background_noise_"]

# Parameters
SR = 16000
DURATION = 1  # seconds per audio chunk
MFCC_FEATURES = 40
MAX_LEN = 32
CONF_THRESHOLD = 0.90  # confidence threshold to detect word

# Queue for audio stream
q = queue.Queue()

# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(indata, frames, time, status):
    """Runs in background to continuously capture audio"""
    if status:
        print(status)
    q.put(indata.copy())

# -----------------------------
# Preprocess audio chunk
# -----------------------------
def preprocess(signal, sr=SR):
    """Convert raw audio to MFCC feature tensor for CNN input."""
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)

    # Pad or truncate to MAX_LEN
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # Add channel and batch dimension for CNN
    return mfcc[..., np.newaxis][np.newaxis, ...]

# -----------------------------
# Main listening loop
# -----------------------------
def listen_and_predict():
    """Continuously listen and predict words from mic."""
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SR, 
                        blocksize=int(SR*DURATION)):
        print("🎤 Listening... (Ctrl+C to stop)")
        while True:
            audio_chunk = q.get().flatten()

            # Preprocess audio
            features = preprocess(audio_chunk)

            # Predict
            pred = model.predict(features, verbose=0)
            idx = np.argmax(pred)
            confidence = np.max(pred)

            # Print detected word if confidence is high
            if confidence >= CONF_THRESHOLD:
                detected_word = WORDS[idx]
                if detected_word not in ["_background_noise_"]:  # ignore background
                    print(f"✅ Detected: {detected_word} (conf={confidence:.2f})")
            else:
                # Optional: uncomment to show "no word detected" feedback
                # print("❌ No valid word detected")
                pass

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    listen_and_predict()

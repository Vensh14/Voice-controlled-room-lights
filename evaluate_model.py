import os
import time
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATASET = "speech_commands_filtered"
WORDS = ["on", "off", "one", "two", "three", "cat", "dog", "wow", "_background_noise_"]
SR, MFCC_FEATURES, MAX_LEN = 16000, 40, 32
MODEL_FILE = "voice_switch_fixed.keras"
CONF_THRESHOLD = 0.90
OUT = "outputs_metrics"
os.makedirs(OUT, exist_ok=True)


def get_mfcc(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=MFCC_FEATURES)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc


# ---- 1. Load data exactly like the training script (same split) ----
print("Loading dataset and extracting MFCCs (this takes a few minutes)...")
X, y = [], []
for label, word in enumerate(WORDS):
    folder = os.path.join(DATASET, word)
    for f in os.listdir(folder):
        if f.lower().endswith(".wav"):
            signal, _ = librosa.load(os.path.join(folder, f), sr=SR)
            X.append(get_mfcc(signal))
            y.append(label)
X = np.array(X)[..., np.newaxis]
y = np.array(y)
print("Dataset:", X.shape, "samples per class:", np.bincount(y))

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- 2. Predict on the test set ----
model = tf.keras.models.load_model(MODEL_FILE, compile=False)
probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(probs, axis=1)
conf = probs.max(axis=1)

names = [w.replace("_background_noise_", "noise") for w in WORDS]
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=names, digits=3)

# accuracy when only confident predictions (>= threshold) are accepted, as in the GUI
mask = conf >= CONF_THRESHOLD
acc_conf = accuracy_score(y_test[mask], y_pred[mask]) if mask.any() else float("nan")
coverage = mask.mean()

# ---- 3. Model size and speed ----
params = model.count_params()
size_mb = os.path.getsize(MODEL_FILE) / 1e6

sample_sig, _ = librosa.load(os.path.join(DATASET, "on", sorted(os.listdir(os.path.join(DATASET, "on")))[0]), sr=SR)
t0 = time.perf_counter()
for _ in range(50):
    get_mfcc(sample_sig)
mfcc_ms = (time.perf_counter() - t0) / 50 * 1000

x1 = X_test[:1].astype("float32")
model(x1, training=False)  # warm-up
t0 = time.perf_counter()
for _ in range(100):
    model(x1, training=False)
cnn_ms = (time.perf_counter() - t0) / 100 * 1000

# ---- 4. Confusion matrix figure ----
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix (test accuracy {acc*100:.1f}%)")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# ---- 5. Print and save a summary ----
summary = f"""
=== Model evaluation ===
Model file            : {MODEL_FILE}
Test samples          : {len(y_test)}  (20% split, random_state=42)
Test accuracy         : {acc*100:.2f}%
Accuracy @ conf>={CONF_THRESHOLD}  : {acc_conf*100:.2f}%  (accepts {coverage*100:.1f}% of samples)
Parameters            : {params:,}
Model file size       : {size_mb:.2f} MB
MFCC extraction time  : {mfcc_ms:.1f} ms per 1 s clip
CNN inference time    : {cnn_ms:.1f} ms per clip (CPU)

=== Per-class metrics ===
{report}
"""
print(summary)
with open(os.path.join(OUT, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(summary)
print("Saved outputs/metrics.txt and outputs/confusion_matrix.png")

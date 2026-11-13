import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

DATASET = "speech_commands_filtered"
WORDS = ["on", "off", "one", "two", "three", "cat", "dog", "wow", "_background_noise_"]
SR, MFCC_FEATURES, MAX_LEN = 16000, 40, 32
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

model = tf.keras.models.load_model("voice_switch_fixed.keras")


def get_mfcc(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=MFCC_FEATURES)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc


def run_cnn(mfcc):
    """Run the model layer by layer, keeping the feature maps after each pooling layer."""
    x = tf.convert_to_tensor(mfcc[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    maps = []
    for layer in model.layers:
        x = layer(x, training=False)
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            maps.append(x.numpy()[0])  # (h, w, channels)
    return x.numpy()[0], maps          # probabilities, feature maps


all_mfcc = {}
for word in WORDS:
    folder = os.path.join(DATASET, word)
    wavs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
    # for the noise folder, skip the README and take the first 1 s of a long recording
    path = os.path.join(folder, wavs[0])
    signal, _ = librosa.load(path, sr=SR, duration=1.0)

    mfcc = get_mfcc(signal)
    all_mfcc[word] = mfcc
    probs, maps = run_cnn(mfcc)
    pred = WORDS[int(np.argmax(probs))]

    print(f"\nTrue: {word:20s} Predicted: {pred:20s} Confidence: {probs.max():.3f}")
    for w, p in zip(WORDS, probs):
        print(f"   {w:20s} {p:.3f}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Word: '{word}'  ->  Predicted: '{pred}' ({probs.max():.2f})", fontsize=14)

    t = np.arange(len(signal)) / SR
    ax[0, 0].plot(t, signal)
    ax[0, 0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

    im = ax[0, 1].imshow(mfcc, origin="lower", aspect="auto", cmap="viridis")
    ax[0, 1].set(title="MFCC (40 x 32)", xlabel="Time frame", ylabel="MFCC coefficient")
    fig.colorbar(im, ax=ax[0, 1])

    for i, fmap in enumerate(maps):
        a = ax[0, 2] if i == 0 else ax[1, 0]
        a.imshow(fmap.mean(axis=-1), origin="lower", aspect="auto", cmap="magma")
        a.set_title(f"CNN block {i + 1} feature map (mean over channels)")

    ax[1, 1].bar(range(len(WORDS)), probs)
    ax[1, 1].set_xticks(range(len(WORDS)))
    ax[1, 1].set_xticklabels([w.replace("_background_noise_", "noise") for w in WORDS], rotation=45)
    ax[1, 1].set(title="CNN output (softmax probabilities)", ylim=(0, 1))

    ax[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"output_{word.strip('_')}.png"), dpi=150)
    plt.close(fig)

# Summary: MFCC of every command side by side
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for a, word in zip(axes.ravel(), WORDS):
    a.imshow(all_mfcc[word], origin="lower", aspect="auto", cmap="viridis")
    a.set_title(word)
fig.suptitle("MFCC for each voice command")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mfcc_all_commands.png"), dpi=150)
print("\nSaved figures in the 'outputs' folder.")
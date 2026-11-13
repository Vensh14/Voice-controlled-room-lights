# ===============================
# Voice Command Recognition - CNN
# Words: on, off, one, two, three, cat, dog, wow, background noise
# ===============================

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

# -------------------------------
# 1. Set dataset path and words
# -------------------------------
DATASET = "speech_commands_filtered"  # path to your filtered dataset
WORDS = ["on", "off", "one", "two", "three", "cat", "dog", "wow", "_background_noise_"]

# -------------------------------
# 2. Parameters
# -------------------------------
SR = 16000         # Sampling rate
MFCC_FEATURES = 40 # Number of MFCC coefficients
MAX_LEN = 32       # Fixed number of time steps for padding/truncation

# -------------------------------
# 3. Load data and extract MFCC sequences
# -------------------------------
X, y = [], []

for label, word in enumerate(WORDS):
    folder = os.path.join(DATASET, word)
    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            path = os.path.join(folder, file)

            # Load audio
            signal, sr = librosa.load(path, sr=SR)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)

            # Pad or truncate MFCCs to a fixed length
            if mfcc.shape[1] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :MAX_LEN]

            X.append(mfcc)
            y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add channel dimension for CNN (samples, features, time, 1)
X = X[..., np.newaxis]

print("Data shape (samples, features, time, channel):", X.shape)
print("Labels shape:", y.shape)

# -------------------------------
# 4. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Build CNN model
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(MFCC_FEATURES, MAX_LEN, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(WORDS), activation='softmax')  # Output = 9 classes
])

# -------------------------------
# 6. Compile model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 7. Train model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,         # You can increase if accuracy is low
    batch_size=32,
    verbose=2
)

# -------------------------------
# 8. Evaluate model
# -------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc:.2f}")

# -------------------------------
# 9. Predict a random test sample
# -------------------------------
idx = random.randint(0, len(X_test)-1)
sample = X_test[idx][np.newaxis, ...]  # Keep batch dimension
pred = model.predict(sample)
print("Predicted:", WORDS[np.argmax(pred)], "| Actual:", WORDS[y_test[idx]])

# -------------------------------
# 10. Save model
# -------------------------------
model.save("voice_switch_fixed.keras")
print("💾 Model saved as voice_switch.h5")

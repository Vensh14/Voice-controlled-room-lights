import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Load one example sound
# -------------------------------
DATASET = "speech_commands_filtered"
word = "on"  # choose a word folder ("on", "off", "one", "two", "three")
file = os.listdir(os.path.join(DATASET, word))[0]  # pick the first file
path = os.path.join(DATASET, word, file)

# Load audio (signal = amplitude over time, sr = sample rate)
signal, sr = librosa.load(path, sr=16000)

# -------------------------------
# 2. Time domain plot
# -------------------------------
plt.figure(figsize=(10, 3))
librosa.display.waveshow(signal, sr=sr)
plt.title(f"Time Domain Waveform - {word}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# -------------------------------
# 3. Frequency domain (spectrum)
# -------------------------------
fft = np.fft.fft(signal)             # Fast Fourier Transform
magnitude = np.abs(fft)              # magnitude of complex numbers
frequency = np.linspace(0, sr, len(magnitude))  # frequency axis

plt.figure(figsize=(10, 3))
plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2])
plt.title(f"Frequency Spectrum - {word}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# -------------------------------
# 4. Spectrogram
# -------------------------------
stft = librosa.stft(signal)          # Short-Time Fourier Transform
spectrogram = np.abs(stft)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                         sr=sr, x_axis="time", y_axis="hz", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title(f"Spectrogram - {word}")
plt.show()

# -------------------------------
# 5. MFCC (like used in model)
# -------------------------------
mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, sr=sr, x_axis="time", cmap="viridis")
plt.colorbar()
plt.title(f"MFCC - {word}")
plt.show()

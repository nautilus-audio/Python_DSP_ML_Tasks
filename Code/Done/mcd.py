'''
A)
Differences in correlation

B) 
Channel Count Mismatch:

Comparing two multi-channel audio files with different numbers of channels (e.g., 5.1 vs. 7.1) leads to misalignment issues.
Solution: Extract and compare corresponding channels (e.g., L & R vs. L & R).
Spatial Information & Panning:

Multi-channel audio has spatialized elements (e.g., background sounds in surround speakers).
Stereo audio collapses spatial data, making it difficult to compare spatial positioning.

Phase and Delay Differences:

Surround mixes often introduce delays between channels to create spatial depth.
This can make cross-correlation alignment difficult since one channel may be offset slightly.

LFE (Low-Frequency Effects) Channel Issues:

Some multi-channel formats contain an LFE (Subwoofer) channel, which does not directly correlate with other channels.
If included in similarity calculations, it may skew results.

C)
Different Mixing Strategies:

A stereo downmix might be derived from a 5.1 mix, but it won’t contain the same spatial characteristics.
Sample Rate Mismatches:

If files have different sample rates, librosa.load(sr=None) ensures they retain native rates, but mismatches may still cause drift over time.
Channel Reordering:

Some multi-channel formats have different channel orders (e.g., WAV vs. AAC). If channels are misaligned, similarity scores will be inaccurate.
Compression & Encoding Differences:

Lossy formats like MP3 or AAC may introduce artifacts that alter the waveform.

'''

import soundfile as sf
import numpy as np
import os
from scipy.signal import correlate
import librosa

def read_audio_files(directory):
    """Read all audio files in a given directory."""
    audio_data = {}
    rates = np.zeros(len(os.listdir(directory)))
    for i, file in enumerate(os.listdir(directory)):
        if file.endswith(('.wav', '.mp3', '.flac')):  # Supported formats
            path = os.path.join(directory, file)
            _, sr = sf.read(path)  # Read audio file
            audio, _ = librosa.load(path, sr=None, mono=False)
            audio_data[i] = audio
            rates[i] = sr

    return audio_data, rates

def compare_audio_files(audio1, audio2, sr1, sr2):
    """Compare two audio files using cross-correlation."""

    # Ensure same sample rate
    if sr1 != sr2:
        raise ValueError("Sample rates do not match.")

    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1, audio2 = audio1[:min_len], audio2[:min_len]

    tempo1 = np.zeros(audio1.shape[0])
    for i in range(audio1.shape[0]):
        tempo, _ = librosa.beat.beat_track(y=audio1[i], sr=sr1)
        tempo1[i] = tempo

    tempo2 = np.zeros(audio2.shape[0])
    for i in range(audio2.shape[0]):
        tempo, _ = librosa.beat.beat_track(y=audio2[i], sr=sr2)
        tempo2[i] = tempo
    
    # Compute cross-correlation
    correlation = np.corrcoef(audio1, audio2)[0, 1]

    return tempo1, tempo2, correlation

if __name__ == "__main__":
    audio_dir = "Audio/Input/FILES_MCD_MULTI"
    data, sr = read_audio_files(audio_dir)
    tempo1, tempo2, similarity = compare_audio_files(data[0], data[1], sr[0], sr[1])
    print(f"Similarity: {similarity}")

    for i in range(data[1].shape[0]):
        print(f"Tempo 1: {tempo1[i]}")
        print(f"Tempo 2: {tempo2[i]}")
"""
Now, apply your method to a new master track using the same stems. Are you 
observing any differences in alignment or gain adjustments? What could be 
causing these variations? How could your approach be adapted to handle it?

    Original Master Results:
    Optimal Time Shift: 155952 samples
    Gain Adjustment forshifted_drums.wav: 0.8410288466335856 dB
    Gain Adjustment forshifted_music.wav: 3.0 dB

    Alt Results:
    Optimal Time Shift: 155241 samples
    Gain Adjustment for shifted_drums.wav: 0.8262289342189924 dB
    Gain Adjustment for shifted_music.wav: 3.0 dB

The time shift and gain adjustments are slightly different when using the
new master track. This could be due to the different characteristics of the
master tracks, such as different mixing strategies, compression, or encoding
differences. To handle these variations, the alignment and gain adjustment
process could be made more robust by incorporating additional features or
machine learning techniques to learn the optimal alignment and gain adjustments
based on the characteristics of the master track and stems.
"""


import numpy as np
import os
import soundfile as sf
from scipy.signal import correlate
import librosa
import pyloudnorm as pyln

def sum_stems(audio_dir):
    """Load and sum stereo audio stems."""
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    data = []

    for file in audio_files:
        file_path = os.path.join(audio_dir, file)
        audio, _ = sf.read(file_path)

        # Ensure all files are stereo
        if audio.ndim == 1:  # Convert mono to stereo by duplicating the channel
            audio = np.stack([audio, audio], axis=1)

        audio = np.clip(audio, -1, 1)
        data.append(audio)

    # Ensure all stems have the same length
    min_length = min(len(audio) for audio in data)
    data = [audio[:min_length, :] for audio in data]  # Trim to shortest file length

    summed_stems = np.sum(data, axis=0)  # Sum along the third dimension (stereo)
    summed_stems = np.clip(summed_stems, -1, 1)  # Prevent clipping after summing
    return summed_stems

def find_shift(signal, reference):
    """Find time shift for each stereo channel separately and take the average shift."""
    shifts = []
    for channel in range(2):  # Left and right channels separately
        cor = correlate(signal[:, channel], reference[:, channel], mode="full")
        optimal_shift = np.argmax(cor) - (len(reference) - 1)
        shifts.append(optimal_shift)
    
    return int(np.mean(shifts))  # Average shift across channels

def apply_time_shift(signal, shift):
    """Applies a time shift to a stereo signal."""
    return np.roll(signal, -shift, axis=0)

def calculate_loudness(file_path):
    """Calculate LUFS and RMS loudness of an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    meter = pyln.Meter(sr)  
    lufs = meter.integrated_loudness(y)
    rms = np.sqrt(np.mean(y ** 2))
    return lufs, rms, sr

def find_gain_adjustment(master_path, stems_dir):
    """Determine gain adjustments for stems to match the master track loudness."""
    master_lufs, master_rms, _ = calculate_loudness(master_path)
    adjustments = {}
    
    for file in os.listdir(stems_dir):
        if file.endswith(".wav"):
            stem_path = os.path.join(stems_dir, file)
            stem_lufs, stem_rms, sr = calculate_loudness(stem_path)

            # Balanced loudness adjustment: blend LUFS and RMS
            lufs_adjustment = master_lufs - stem_lufs
            rms_adjustment = 20 * np.log10(master_rms / (stem_rms + 1e-6))  # Avoid division by zero
            gain_adjustment = (lufs_adjustment * 0.7) + (rms_adjustment * 0.3)

            # Cap excessive gain changes
            gain_adjustment = np.clip(gain_adjustment, -6, 3)  # Max boost +3 dB, max cut -6 dB

            adjustments[file] = (gain_adjustment, sr)

    return adjustments

def apply_gain_adjustment(data, output_path, gain_db, sr):
    """Apply a gain adjustment to an audio file and save the result."""
    y = data
    gain_factor = 10 ** (gain_db / 20)  # Convert dB to linear scale
    y_adjusted = np.clip((y * gain_factor) * .707, -1, 1)  # Prevent clipping
    sf.write(output_path, y_adjusted, sr)
    return y_adjusted

if __name__ == "__main__":
    stems_dir = "Audio/Input/FILES_GSAS_STEMS_MASTER/shifted_stems/"
    master_path = "Audio/Input/FILES_GSAS_STEMS_MASTER/master.wav"
    summed_stems = sum_stems(stems_dir)
    
    master_track, sr = sf.read(master_path)

    audio_files = [f for f in os.listdir(stems_dir) if f.endswith(".wav")]

    # Ensure master is stereo
    if master_track.ndim == 1:
        master_track = np.stack([master_track, master_track], axis=1)

    # Find time shift
    shift_value = find_shift(summed_stems, master_track)
    print(f"Optimal Time Shift: {shift_value} samples")
    num_stems = len(audio_files)
    data = np.zeros((4, 899178, 2))

    # Apply shift and gain to stereo stems
    for i, file in enumerate(audio_files):
        file_path = os.path.join(stems_dir, file)
        audio, _ = sf.read(file_path)
        file_name = file.split(".")[0]
        gains = find_gain_adjustment(master_path, stems_dir)
        print(f"Gain Adjustment for {file}: {gains[file][0]} dB")
        shifted_stem = apply_time_shift(audio, shift_value)
        data[i:i+2, :, :] = apply_gain_adjustment(shifted_stem, f"Audio/Output/GSAS_Stems/{file_name}_Adj.wav", gains[file][0], sr)

    # Save the adjusted summed stems
    adj_stems_dir = "Audio/Output/GSAS_Stems"
    summed_stems_adjusted = sum_stems(adj_stems_dir)
    sf.write("Audio/Output/GSAS_Stems/summed_stems_adjusted.wav", summed_stems_adjusted, sr)

import soundfile as sf
import numpy as np

# Define the 7.1.4 channel layout order (ITU standard)
CHANNEL_LAYOUT_7_1_4 = [
    "L", "R", "C", "LFE", "Ls", "Rs", "Lb", "Rb",  # 7.1 base
    "Ltf", "Rtf", "Ltr", "Rtr"                    # 4 height channels
]

def load_bwf_file(file_path):
    """Loads a multi-channel BWF file."""
    audio, sample_rate = sf.read(file_path)
    print(sf.info(file_path))

    if audio.shape[1] < 22:
        raise ValueError("Expected at least 22 channels in the BWF file.")
    return audio, sample_rate

def project_to_7_1_4(audio_22ch):
    """Maps the 22-channel audio onto a 7.1.4 speaker layout."""
    
    if audio_22ch.shape[1] < 22:
        raise ValueError("Input audio must have at least 22 channels.")
    
    num_samples = audio_22ch.shape[0]
    
    # Create an empty 7.1.4 buffer
    audio_7_1_4 = np.zeros((num_samples, len(CHANNEL_LAYOUT_7_1_4)))

    # Direct mappings
    audio_7_1_4[:, 0] = audio_22ch[:, 0]   # L -> Channel 1
    audio_7_1_4[:, 1] = audio_22ch[:, 1]   # R -> Channel 2
    audio_7_1_4[:, 2] = audio_22ch[:, 2]   # C -> Channel 3
    audio_7_1_4[:, 3] = audio_22ch[:, 3]   # LFE -> Channel 4
    audio_7_1_4[:, 4] = audio_22ch[:, 4]   # Ls -> Channel 5
    audio_7_1_4[:, 5] = audio_22ch[:, 5]   # Rs -> Channel 6
    audio_7_1_4[:, 6] = audio_22ch[:, 6]   # Lb -> Channel 7
    audio_7_1_4[:, 7] = audio_22ch[:, 7]   # Rb -> Channel 8

    # Height channels (combine channels via summation or averaging as needed)
    audio_7_1_4[:, 8]  = np.clip(audio_22ch[:, 8] + audio_22ch[:, 9], -1.0, 1.0)  # Ltf (Left Top Front)  = Channel 9 + 10
    audio_7_1_4[:, 9]  = np.clip(audio_22ch[:, 10] + audio_22ch[:, 11], -1.0, 1.0) # Rtf (Right Top Front) = Channel 11 + 12
    audio_7_1_4[:, 10] = np.clip(audio_22ch[:, 12] + audio_22ch[:, 13], -1.0, 1.0) # Ltr (Left Top Rear) = Channel 13 + 14
    audio_7_1_4[:, 11] = np.clip(audio_22ch[:, 14] + audio_22ch[:, 15], -1.0, 1.0) # Rtr (Right Top Rear) = Channel 15 + 16

    # Normalize if necessary
    max_val = np.max(np.abs(audio_7_1_4))
    if max_val > 0:
        audio_7_1_4 /= max_val  # Normalize to prevent low output volume

    return audio_7_1_4

def load_wav_file(file_path):
    """Loads a multi-channel WAV file."""
    audio, sample_rate = sf.read(file_path)
    return audio, sample_rate

def save_bwf_file(output_path, audio_data, sample_rate):
    """Saves the processed 7.1.4 audio file in BWF format."""
    sf.write(output_path, audio_data, sample_rate, format="WAV", subtype="PCM_24")
    print(f"Saved projected 7.1.4 BWF: {output_path}")

if __name__ == "__main__":
    input_file = "Audio/Input/BWF/STAFF_TEST_ATMOS_MASTER.wav"
    output_file = "Audio/Output/output_7_1_4.wav"
    reference_file = "Audio/Input/BWF/STAFF_TEST_ATMOS_MASTER_7.1.4.wav"

     # Load reference
    audio_22ch, sr = load_bwf_file(input_file)

    # Load the original 22-channel BWF
    reference, sr = load_wav_file(reference_file)

    # Project to 7.1.4
    audio_7_1_4 = project_to_7_1_4(audio_22ch)

    # Save the remapped audio
    save_bwf_file(output_file, audio_7_1_4, sr)

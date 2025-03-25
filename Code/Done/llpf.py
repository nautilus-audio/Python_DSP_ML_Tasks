"""
1. How many lines of code would you need to do a linear-phase low-pass filter?
5 lines of code. 

2. What do “poles” mean in relation to a filter, like the one you built in (1)?
Poles are the roots of the denominator polynomial of the transfer function of a filter. 
They determine the stability and frequency response of the filter.

3. Why is oversampling useful?
Oversampling is useful for reducing aliasing artifacts when filtering a signal.
"""


import numpy as np
from scipy.signal import lfilter, firwin
import soundfile as sf
import os

def oversample_signal(signal, factor):
    """Upsample a signal by inserting zeros between samples."""
    upsampled = np.zeros(len(signal) * factor)
    upsampled[::factor] = signal  # Place original samples, insert zeros in between
    return upsampled

def downsample_signal(signal, factor):
    """Downsample a signal by taking every nth sample."""
    return signal[::factor]

def fir_lowpass_filter(input, cutoff, num_taps, sample_rate):
    """Design a lowpass FIR filter."""
    fir_coeff = firwin(num_taps, cutoff, fs=sample_rate, window="hamming")
    return lfilter(fir_coeff, 1.0, input)

def main():
    """Main function demonstrating oversampling, filtering, and downsampling."""
    # Generate a test signal (1 kHz sine wave)
    fs = 8000  # Original sampling rate
    duration = 2.0  # 10ms
    t = np.arange(0, duration, 1/fs)
    input_signal = np.sin(2 * np.pi * 1000 * t)  # 1 kHz sine wave

    sf.write("Audio/Input/sineIn.wav", input_signal, fs)

    output_dir = "Audio/Output/"

    if not os.path.isdir(output_dir):  # Correct way to check if directory does not exist
        os.mkdir(output_dir)  # Create the directory

    # Oversampling
    oversampling_factor = 4
    fs_oversampled = fs * oversampling_factor
    oversampled_signal = oversample_signal(input_signal, oversampling_factor)

    # Lowpass filtering (anti-aliasing filter)
    cutoff_freq = fs / 2  # Nyquist frequency
    num_taps = 101  # Filter order
    filtered_signal = fir_lowpass_filter(oversampled_signal, cutoff_freq, num_taps, fs_oversampled)

    # Downsampling back to original rate
    downsampled_signal = downsample_signal(filtered_signal, oversampling_factor)
    output_filename = f"Audio/Output/sineOut.wav"
    sf.write(output_filename, downsampled_signal, fs)

if __name__ == "__main__":
    main()



"""


1) Feature extraction: extract relevant features from the audio, such as: 
Spectral features (centroid, bandwidth, rolloff), which are useful for detecting processed audio.
Transient detection (onset_env), which helps distinguish between transient (percussive) and sustained sounds.
Zero Crossing Rate (ZCR), which can also help identify percussive sounds.

2) Train a machine learning model to classify the audio as processed or unprocessed, melodic or rhythmic.

To detect and qualify the function of an instrument (e.g., determining whether an instrument is playing a 
melodic line or a rhythmic part):

You can use feature-based classification to detect rhythmic versus melodic elements based on spectral content, transients, and timing.
The current features extracted (spectral centroid, bandwidth, rolloff) are good indicators of whether a sound is more percussive (sharp transients, high-frequency content) or sustained (melodic).
For example:

Percussive sounds: Will have high onset strength and zero-crossing rate values due to their sharp, short bursts.
Musical (sustained) sounds: Will have more stable spectral centroid values and lower onset strength.

3)
Spatial and Prominence Analysis: Based on RMS and stereo balance, you can begin to infer:

The prominence of each instrument by comparing RMS values across the channels.
The spatial placement of the instrument by comparing left/right panning features, and if available, reverberation time for front-back placement.

"""
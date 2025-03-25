import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(file_path):
    """Extracts relevant audio features for classification."""
    y, sr = librosa.load(file_path, sr=16000)  # Load the audio at a fixed sampling rate

    # Extract MFCC (Mel Frequency Cepstral Coefficients) as features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Take the mean of each MFCC coefficient across time
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean

# ==============================
# TRAIN THE MODEL
# ==============================
def train_model(audio_dir):
    """Loads audio files, extracts features, and trains a RandomForest classifier."""
    X = []  # Feature vectors
    y = []  # Labels (0: vocals, 1: drums, 2: bass, 3: other)

    # Loop through all audio files in the specified directory
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            # Extract features from the audio file
            file_path = os.path.join(audio_dir, file)
            features = extract_features(file_path)
            X.append(features)

            # Define labels based on file naming convention or manually
            if 'vocals' in file:
                label = 0  # vocals
            elif 'drums' in file:
                label = 1  # drums
            elif 'bass' in file:
                label = 2  # bass
            else:
                label = 3  # other
            y.append(label)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(clf, "audio_classifier.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # Evaluate the model on the test set
    accuracy = clf.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")

# ==============================
# CLASSIFY A NEW AUDIO FILE
# ==============================
def classify_audio(file_path, model_path="audio_classifier.pkl", scaler_path="scaler.pkl"):
    """Classifies an audio file into vocals, drums, bass, or other."""
    clf = joblib.load(model_path)  # Load the trained model
    scaler = joblib.load(scaler_path)  # Load the scaler used during training

    # Extract features from the input file
    features = extract_features(file_path).reshape(1, -1)

    # Standardize the features using the same scaler
    features = scaler.transform(features)

    # Predict the class label
    prediction = clf.predict(features)

    # Map the class label to a category
    categories = ['vocals', 'drums', 'bass', 'other']
    result = categories[prediction[0]]

    print(f"The audio '{file_path}' is classified as: {result}")

# ==============================
# MAIN FUNCTION
# ==============================
# ==============================
# MAIN FUNCTION
# ==============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Classification Script")
    parser.add_argument("mode", choices=["train", "predict"], help="Choose whether to train the model or predict the class of an audio file.")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files for training.")
    parser.add_argument("--file", type=str, help="Path to an audio file for prediction.")

    args = parser.parse_args()

    # Handle the training mode
    if args.mode == "train":
        if not args.audio_dir:
            print("Error: Please provide a directory with audio files for training using --audio_dir")
        else:
            print("Training the model with audio files in:", args.audio_dir)
            train_model(args.audio_dir)
    
    # Handle the prediction mode
    elif args.mode == "predict":
        if not args.file:
            print("Error: Please provide an audio file for prediction using --file")
        else:
            print("Classifying the audio file:", args.file)
            classify_audio(args.file)


import joblib
import os
import pandas as pd
import time
import random
import pygame
from modules.feature_extraction import extract_features
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import json  # For logging metrics

# Loading the trained model that we created earlier (using SVM on EEG features)
MODEL_PATH = "models/best_model.pkl"

# This is the EEG data CSV file I'm using for testing the system right now
CSV_PATH = "datasets/kaggle_temp/emotion.csv"

# Here I’m linking each predicted emotion to a folder where I’ll keep related songs
MUSIC_FOLDER_MAP = {
    "POSITIVE": "music/positive",
    "NEGATIVE": "music/negative",
    "NEUTRAL": "music/neutral"
}

# This is the time delay between emotion checks (like simulating real-time EEG input)
CHECK_INTERVAL = 5

# Metrics logging path (for analysis and paper writing)
METRICS_LOG_PATH = "results/metrics_log.json"

# Loading the trained model from file
print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# Just assigning index values to each emotion to help while plotting later
EMOTION_INDEX = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

# Initializing pygame so we can play music files
pygame.mixer.init()

# Setting up a live bar graph to show which emotion is being predicted in real time
plt.ion()
fig, ax = plt.subplots()
emotion_history = deque(maxlen=30)  # Keeps track of recent predictions

# Setting up a bar chart with 3 bars for NEGATIVE, NEUTRAL, and POSITIVE
bar = ax.bar(["NEGATIVE", "NEUTRAL", "POSITIVE"], [0, 0, 0], color=['red', 'gray', 'green'])
ax.set_ylim(0, 1)
ax.set_title("Real-time Emotion Prediction")

# These two will track the currently detected emotion and currently playing song
current_emotion = None
current_song = None

# To store metrics
metrics_log = []

# This function plays a random MP3 file from the emotion-specific folder
def play_random_music(folder_path):
    global current_song
    songs = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]

    if not songs:
        print(f"No songs found in {folder_path}")
        return

    selected = random.choice(songs)
    song_path = os.path.join(folder_path, selected)

    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

    print(f" Playing: {selected}")
    current_song = selected

# This function reads the EEG CSV file in small chunks — 
# it's like pretending we are getting real-time EEG input
def get_chunk_generator(csv_path, chunk_size=1):
    df = pd.read_csv(csv_path)
    total_rows = df.shape[0]

    for i in range(0, total_rows, chunk_size):
        yield df.iloc[i:i+chunk_size]

# Create the generator that we’ll use to simulate EEG input
chunk_generator = get_chunk_generator(CSV_PATH, chunk_size=1)

# This is the main loop that runs the whole emotion → music logic
try:
    for chunk in chunk_generator:
        # Step 1: Extract features from this 1-row EEG chunk
        features = extract_features(chunk, dataset_format='custom')

        if features.shape[0] == 0:
            print(" No valid features extracted from chunk.")
            continue

        # Step 2: Predict the emotion using the model
        pred = model.predict(features)
        emotion = pred[0]
        print(f" Predicted Emotion: {emotion}")

        # Log prediction with timestamp
        metrics_log.append({
            "timestamp": time.time(),
            "emotion": emotion
        })

        # Step 3: If emotion changed from previous, play new song accordingly
        if emotion != current_emotion:
            current_emotion = emotion
            folder = MUSIC_FOLDER_MAP.get(emotion)
            if folder:
                play_random_music(folder)
            else:
                print(f" No folder mapped for emotion: {emotion}")
        else:
            # If emotion is same, but the song has stopped, play another one
            if not pygame.mixer.music.get_busy():
                folder = MUSIC_FOLDER_MAP.get(emotion)
                if folder:
                    play_random_music(folder)

        # Step 4: Update the live bar graph with current emotion
        for rect in bar:
            rect.set_height(0)
        if emotion in EMOTION_INDEX:
            bar[EMOTION_INDEX[emotion]].set_height(1)

        plt.draw()
        plt.pause(0.01)

        # Step 5: Just showing emotion in terminal nicely
        tqdm.write(f" Emotion: {emotion}")

        # Simulate delay like we're getting live EEG every few seconds
        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    # If I press Ctrl+C to stop, this will stop the music too
    print("\n Stopped by user. Shutting down...")
    pygame.mixer.music.stop()

    # Save metrics log to file for further analysis
    with open(METRICS_LOG_PATH, "w") as f:
        json.dump(metrics_log, f, indent=4)
    print(f" Metrics logged to {METRICS_LOG_PATH}")

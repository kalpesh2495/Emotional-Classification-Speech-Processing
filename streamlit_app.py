import streamlit as st
import numpy as np
import librosa
import joblib

# Set page title
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🎤 Speech Emotion Recognition")
st.write("**Note:** TensorFlow model temporarily disabled due to compatibility issues.")
st.write("This is a demo version showing the audio processing pipeline.")

# Feature extraction function (same as before)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3.0, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feats = [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]
        feats.append(librosa.feature.chroma_stft(y=y, sr=sr))
        rmse = librosa.feature.rms(y=y)
        feats.append(np.repeat(rmse, mfcc.shape[1] // rmse.shape[1] + 1, axis=1)[:, :mfcc.shape[1]])
        full = np.vstack(feats)
        full = (full - full.mean()) / (full.std() + 1e-6)
        full = full.T
        full = np.pad(full, ((0, max(0, 173 - full.shape[0])), (0, 0)), mode='constant')[:173]
        return full
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Streamlit UI
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.success("File uploaded successfully! 🎉")
    
    # Save the uploaded file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    # Extract features
    with st.spinner("Processing audio..."):
        features = extract_features("temp.wav")
    
    if features is not None:
        st.success("✅ Audio features extracted successfully!")
        st.write(f"**Feature shape:** {features.shape}")
        
        # Show some basic audio info
        y, sr = librosa.load("temp.wav", duration=3.0, offset=0.5)
        st.write(f"**Audio duration:** {len(y)/sr:.2f} seconds")
        st.write(f"**Sample rate:** {sr} Hz")
        
        # For now, just show a random prediction (replace this when TensorFlow works)
        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
        predicted_emotion = np.random.choice(emotions)
        
        st.info(f"🎭 **Demo Prediction:** {predicted_emotion}")
        st.write("*(This is a random prediction for demo purposes)*")
        
        # Display audio
        st.audio("temp.wav")
    
    # Clean up
    import os
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")

st.markdown("---")
st.markdown("**Status:** Working on TensorFlow compatibility. Audio processing is functional! 🎵")

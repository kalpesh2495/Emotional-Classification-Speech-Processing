import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input

# Set page title
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

# Load model architecture
def residual_block(x, filters, kernel_size, pool_size, dropout):
    skip = x
    x = L.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling1D(pool_size, padding='same')(x)
    x = L.Dropout(dropout)(x)
    skip = L.Conv1D(filters, 1, padding='same')(skip)
    skip = L.MaxPooling1D(pool_size, padding='same')(skip)
    return L.add([x, skip])

def build_model(input_shape, n_classes):
    inp = Input(shape=input_shape)
    x = residual_block(inp, 256, 5, 2, 0.2)
    x = residual_block(x,   256, 5, 2, 0.2)
    x = residual_block(x,   128, 3, 2, 0.2)
    x = L.Bidirectional(L.GRU(128, return_sequences=True))(x)
    x = L.Dropout(0.3)(x)
    scores = L.Dense(64, activation='tanh')(x)
    scores = L.Dense(1)(scores)
    weights = L.Softmax(axis=1)(scores)
    x = L.multiply([x, weights])
    x = L.Lambda(lambda z: tf.reduce_sum(z, axis=1), output_shape=(256,))(x)
    x = L.Dense(256, activation='relu')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation='softmax')(x)
    return Model(inp, out)

# Load model and label encoder
model = build_model((173, 133), n_classes=7)
model.load_weights("emotion_model_no_neutral.h5")
le = joblib.load("label_encoder_no_neutral.pkl")

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3.0, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats = [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]
    feats.append(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = librosa.feature.rms(y=y)
    feats.append(np.repeat(rmse, mfcc.shape[1] // rmse.shape[1], axis=1))
    full = np.vstack(feats)
    full = (full - full.mean()) / (full.std() + 1e-6)
    full = full.T
    full = np.pad(full, ((0, max(0, 173 - full.shape[0])), (0, 0)), mode='constant')[:173]
    return np.expand_dims(full, axis=0)

# Streamlit UI
st.title("🎤 Speech Emotion Recognition")
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    features = extract_features("temp.wav")
    preds = model.predict(features)
    pred_label = le.inverse_transform([np.argmax(preds)])
    st.success(f"🎧 Predicted Emotion: {pred_label[0]}")
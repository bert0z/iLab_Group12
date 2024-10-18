import streamlit as st
import xgboost as xgb
import librosa
import parselmouth
import numpy as np
import pandas as pd
from joblib import load
import nolds
import antropy as ant

pd_model = load('models/Custom_model.joblib')

st.title("Parkinson's Prediction from Voice")

def extract_features(audio_path):
    # Load audio
    sound = parselmouth.Sound(audio_path)
    y, sr = librosa.load(audio_path)

    # 1. MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
    pitch = sound.to_pitch()
    fo = pitch.selected_array['frequency']
    MDVP_Fo = np.mean(fo[fo > 0])  # Mean fundamental frequency
    MDVP_Fhi = np.max(fo)           # Max fundamental frequency
    MDVP_Flo = np.min(fo[fo > 0])   # Min fundamental frequency

    # 2. NHR and HNR
    harmonicity = sound.to_harmonicity()
    HNR = harmonicity.values[harmonicity.values != -200].mean()
    NHR = 1 / HNR if HNR != 0 else 0

    # 3. RPDE (Recurrence Period Density Entropy)
    RPDE = ant.perm_entropy(fo, normalize=True)    # or use a different signal for better accuracy

    # 4. DFA (Detrended Fluctuation Analysis)
    DFA = nolds.dfa(y)

    # 5. Spread2
    spread = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    Spread2 = spread.mean() / (sr / 2)

    # 6. D2 (Correlation Dimension)
    D2 = nolds.corr_dim(fo, emb_dim=25)  # Adjust emb_dim for accuracy, perhaps check paper

    # 7. PPE (Pitch Period Entropy)
    PPE = ant.app_entropy(fo)

    # Consolidate into dictionary
    features = {
        "MDVP:Fo(Hz)": MDVP_Fo,
        "MDVP:Fhi(Hz)": MDVP_Fhi,
        "MDVP:Flo(Hz)": MDVP_Flo,
        "NHR": NHR,
        "HNR": HNR,
        "RPDE": RPDE,
        "DFA": DFA,
        "Spread2": Spread2,
        "D2": D2,
        "PPE": PPE
    }
    
    return features

uploaded_file = st.file_uploader("Upload a .wav file", type="wav")
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the uploaded file
    features = extract_features("temp.wav")
    
    # Convert features to DataFrame format
    feature_df = pd.DataFrame([features])

    # Make prediction
    prediction = pd_model.predict(feature_df)
    prediction_text = "Signs of Parkinson's" if prediction[0] == 1 else "Healthy"

    # Show result
    st.write("Prediction:", prediction_text)
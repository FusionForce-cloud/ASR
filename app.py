import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import librosa

st.title("Telugu ASR Inference Demo")

@st.cache(allow_output_mutation=True)
def load_model():
    proc = Wav2Vec2Processor.from_pretrained("wav2vec2_telugu")
    mod = Wav2Vec2ForCTC.from_pretrained("wav2vec2_telugu")
    return proc, mod

processor, model = load_model()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    # Save uploaded bytes to temp WAV file
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    speech, sr = sf.read("temp.wav")
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    st.audio(audio_bytes, format='audio/wav')
    # Prepare input and run model
    input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids[0], skip_special_tokens=True)
    transcription = transcription.replace("|", " ")
    st.write("**Transcription:** " + transcription)

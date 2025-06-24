import os
import wave
import pyaudio
import json
import uuid
import google.generativeai as genai
from groq import Groq
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


# Flask App Initialization
app = Flask(__name__)
CORS(app)

# API Keys
GROQ_API_KEY= "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"
GEMINI_API_KEY= "AIzaSyDqv1F3J6cb6BIT2Y_fghQRksBdxvJ8E-0"

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 6
AUDIO_FILENAME = "audio.wav"
CHAT_AUDIO_FILENAME = "chat_audio.wav"

# CSV File for Data Storage
CSV_FILE = "patient_data.csv"

# Ensure CSV File Exists
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["UUID", "Audio Input", "Summary", "Patient Name", "Age", "Gender",
                               "Estimated Disease", "Symptoms", "Patient History",
                               "Date of Diagnosis", "Timestamp"])
    df.to_csv(CSV_FILE, index=False)

### AUDIO RECORDING FUNCTION ###
def record_audio(filename, duration=RECORD_SECONDS):
    """Records audio from the microphone and saves it as a .wav file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

### AUDIO TRANSCRIPTION FUNCTION ###
def transcribe_audio(filename):
    """Uses Whisper (Groq) to transcribe audio."""
    with open(filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        return transcription.text.strip()

### SUMMARIZATION FUNCTION ###
def summarize_text(text):
    """Summarizes transcribed text & extracts patient details using Gemini AI."""
    prompt = f"""
    Summarize the following medical text and extract patient details:
    - Patient Name
    - Age
    - Gender
    - Estimated Disease
    - Symptoms
    - Patient History
    \n\n{text}
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

### EXTRACT PATIENT DETAILS ###
def extract_patient_details(summary):
    patient_name, age, gender = "Unknown", "Unknown", "Unknown"
    estimated_disease, symptoms, patient_history = "Unknown", "Unknown", "Unknown"
    
    lines = summary.split("\n")
    for line in lines:
        if "Patient Name:" in line:
            patient_name = line.split(":", 1)[-1].strip()
        elif "Age:" in line:
            age = line.split(":", 1)[-1].strip()
        elif "Gender:" in line:
            gender = line.split(":", 1)[-1].strip()
        elif "Estimated Disease:" in line:
            estimated_disease = line.split(":", 1)[-1].strip()
        elif "Symptoms:" in line:
            symptoms = line.split(":", 1)[-1].strip()
        elif "Patient History:" in line:
            patient_history = line.split(":", 1)[-1].strip()

    return patient_name, age, gender, estimated_disease, symptoms, patient_history

### SAVE TO CSV ###
def save_to_csv(audio_text, summary):
    """Saves patient details into CSV."""
    patient_name, age, gender, estimated_disease, symptoms, patient_history = extract_patient_details(summary)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    patient_uuid = str(uuid.uuid4())

    df = pd.read_csv(CSV_FILE)
    new_entry = pd.DataFrame([{
        "UUID": patient_uuid,
        "Audio Input": audio_text,
        "Summary": summary,
        "Patient Name": patient_name,
        "Age": age,
        "Gender": gender,
        "Estimated Disease": estimated_disease,
        "Symptoms": symptoms,
        "Patient History": patient_history,
        "Date of Diagnosis": timestamp.split(" ")[0],
        "Timestamp": timestamp
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return {
        "uuid": patient_uuid,
        "name": patient_name,
        "age": age,
        "gender": gender,
        "disease": estimated_disease,
        "symptoms": symptoms
    }

### FLASK ROUTES ###
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/record_patient', methods=['POST'])
def record_patient():
    record_audio(AUDIO_FILENAME)
    transcribed_text = transcribe_audio(AUDIO_FILENAME)
    summary = summarize_text(transcribed_text)
    data = save_to_csv(transcribed_text, summary)
    return jsonify(data)

@app.route('/record_chat', methods=['POST'])
def record_chat():
    record_audio(CHAT_AUDIO_FILENAME, duration=5)
    user_question = transcribe_audio(CHAT_AUDIO_FILENAME)

    df = pd.read_csv(CSV_FILE)
    system_prompt = "You are a helpful AI medical assistant."
    response = generate_response(system_prompt, user_question, df)

    return jsonify({"question": user_question, "answer": response})

@app.route('/chat_text', methods=['POST'])
def chat_text():
    user_input = request.json.get("message", "")

    df = pd.read_csv(CSV_FILE)
    system_prompt = "You are a helpful AI medical assistant."
    response = generate_response(system_prompt, user_input, df)

    return jsonify({"answer": response})

def generate_response(system_prompt, user_question, df):
    """Finds relevant patient data before sending it to AI."""
    
    # Filter data based on keywords in the user's question
    relevant_data = df[df.apply(lambda row: any(keyword.lower() in row.to_string().lower() for keyword in user_question.split()), axis=1)]

    # If no relevant data is found, just use the last 5 records
    if relevant_data.empty:
        relevant_data = df.tail(3)

    data_context = relevant_data.to_string(index=False)  # Reduce dataset size

    full_prompt = f"""
    {system_prompt}

    Relevant Patient Info:{relevant_data[['Patient Name', 'Age', 'Estimated Disease', 'Symptoms']].to_string(index=False)}


    Question: {user_question}
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": full_prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(debug=True)

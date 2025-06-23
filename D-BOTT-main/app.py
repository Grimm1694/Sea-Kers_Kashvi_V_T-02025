
import os
import wave
import pyaudio
import json
import uuid
import google.generativeai as genai
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Flask App Initialization
app = Flask(__name__)
CORS(app, origins=["http://localhost:5001", "http://127.0.0.1:5500"])

# API Keys
GROQ_API_KEY= "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"
GEMINI_API_KEY= "AIzaSyAfMEqBRL3U0ez64jRTuL6YTn636SqLsZo"
MICROPHONE_INDEX = 1  # Default to system default

if not GROQ_API_KEY or not GEMINI_API_KEY:
    logging.error("API keys not found in .env file")
    raise ValueError("API keys not found in .env file")

# Validate API Keys
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    # Test Groq API with a simple request (e.g., list models)
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10
    )
    logging.info("Groq API key validated successfully")
except Exception as e:
    logging.error(f"Groq API key validation failed: {e}")
    raise ValueError("Invalid Groq API key")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    test_response = gemini_model.generate_content("Test")
    logging.info("Gemini API key validated successfully")
except Exception as e:
    logging.error(f"Gemini API key validation failed: {e}")
    raise ValueError("Invalid Gemini API key")

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 7
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
    logging.info(f"Created CSV file: {CSV_FILE}")

def record_audio(filename, duration=RECORD_SECONDS):
    """Records audio from the microphone and saves it as a .wav file."""
    try:
        audio = pyaudio.PyAudio()
        device_count = audio.get_device_count()
        if device_count == 0:
            logging.error("No audio input devices found")
            raise RuntimeError("No audio input devices found")
        
        logging.info(f"Found {device_count} audio devices:")
        for i in range(device_count):
            device_info = audio.get_device_info_by_index(i)
            logging.info(f"Device {i}: {device_info['name']}, Input Channels: {device_info['maxInputChannels']}")
        
        device_index = MICROPHONE_INDEX if MICROPHONE_INDEX >= 0 else audio.get_default_input_device_info()['index']
        device_info = audio.get_device_info_by_index(device_index)
        if device_info['maxInputChannels'] == 0:
            logging.error(f"Selected device {device_index} ({device_info['name']}) has no input channels")
            raise RuntimeError("Selected device has no input channels")
        
        logging.info(f"Using device {device_index}: {device_info['name']}")
        
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            input_device_index=device_index,
                            frames_per_buffer=CHUNK)
        logging.info("Recording audio...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        file_size = os.path.getsize(filename)
        logging.info(f"Saved audio to {filename} (size: {file_size} bytes)")
        
        # Check audio amplitude
        with wave.open(filename, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
            max_amplitude = np.max(np.abs(samples))
            logging.info(f"Audio max amplitude: {max_amplitude}")
            if max_amplitude < 200:  # Increased threshold for silence
                logging.warning(f"Audio file {filename} may be silent (max amplitude: {max_amplitude})")
        
        if file_size < 1000:
            logging.warning(f"Audio file {filename} may be empty (size: {file_size} bytes)")
    except Exception as e:
        logging.error(f"Audio recording error: {e}")
        raise
    finally:
        if 'audio' in locals():
            audio.terminate()

def transcribe_audio(filename):
    """Transcribes audio using Groq's Whisper model."""
    try:
        if not os.path.exists(filename):
            logging.error(f"Audio file not found: {filename}")
            raise FileNotFoundError(f"Audio file not found: {filename}")
        file_size = os.path.getsize(filename)
        if file_size < 1000:
            logging.error(f"Audio file {filename} is too small (size: {file_size} bytes)")
            raise ValueError(f"Audio file is too small")
        with open(filename, "rb") as file:
            try:
                transcription = groq_client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json"
                )
                logging.info(f"Transcription: {transcription.text}")
                if not transcription.text.strip():
                    logging.warning(f"Empty transcription for {filename}")
                return transcription.text.strip()
            except requests.exceptions.HTTPError as http_err:
                logging.error(f"Groq API HTTP error: {http_err}")
                raise
            except Exception as api_err:
                logging.error(f"Groq API error: {api_err}")
                raise
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        raise

def summarize_text(text):
    """Summarizes transcribed text & extracts patient details using Gemini AI."""
    try:
        if not text.strip():
            logging.warning("Empty text provided for summarization")
            return "No valid text provided"
        prompt = f"""
        Summarize the following medical text and extract patient details in this exact format:
        - Patient Name: [Name]
        - Age: [Age]
        - Gender: [Gender]
        - Estimated Disease: [Disease]
        - Symptoms: [Symptoms]
        - Patient History: [History]
        \n\n{text}
        """
        response = gemini_model.generate_content(prompt)
        logging.info(f"Summary: {response.text}")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        raise

def extract_patient_details(summary):
    """Extracts patient details from the summary text."""
    patient_name, age, gender = "Unknown", "Unknown", "Unknown"
    estimated_disease, symptoms, patient_history = "Unknown", "Unknown", "Unknown"
    try:
        lines = summary.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Patient Name:"):
                patient_name = line.split(":", 1)[-1].strip() or "Unknown"
            elif line.startswith("Age:"):
                age = line.split(":", 1)[-1].strip() or "Unknown"
            elif line.startswith("Gender:"):
                gender = line.split(":", 1)[-1].strip() or "Unknown"
            elif line.startswith("Estimated Disease:"):
                estimated_disease = line.split(":", 1)[-1].strip() or "Unknown"
            elif line.startswith("Symptoms:"):
                symptoms = line.split(":", 1)[-1].strip() or "Unknown"
            elif line.startswith("Patient History:"):
                patient_history = line.split(":", 1)[-1].strip() or "Unknown"
        logging.info(f"Extracted details: Name={patient_name}, Age={age}, Gender={gender}, "
                     f"Disease={estimated_disease}, Symptoms={symptoms}, History={patient_history}")
    except Exception as e:
        logging.error(f"Error extracting patient details: {e}")
    return patient_name, age, gender, estimated_disease, symptoms, patient_history

def save_to_csv(audio_text, summary):
    """Saves patient details into CSV."""
    try:
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
        logging.info(f"Saved to CSV: UUID={patient_uuid}")
        return {
            "uuid": patient_uuid,
            "name": patient_name,
            "age": age,
            "gender": gender,
            "disease": estimated_disease,
            "symptoms": symptoms
        }
    except Exception as e:
        logging.error(f"CSV save error: {e}")
        raise

def generate_response(system_prompt, user_question, df):
    """Generates a response using Groq's API with relevant patient data."""
    try:
        relevant_data = df[df.apply(
            lambda row: any(keyword.lower() in row.to_string().lower() 
                            for keyword in user_question.split()), 
            axis=1)]
        if relevant_data.empty:
            relevant_data = df.tail(3)
        full_prompt = f"""
        {system_prompt}

        Relevant Patient Info:
        {relevant_data[['Patient Name', 'Age', 'Gender', 'Symptoms']].to_string(index=False)}

        Question: {user_question}
        """
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )
        logging.info(f"Generated response: {response.choices[0].message.content}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        raise

@app.route('/')
def home():
    logging.info("Serving index.html")
    return render_template("index.html")

@app.route('/record_patient', methods=['POST'], strict_slashes=False)
def record_patient():
    logging.info(f"Received POST request to /record_patient from {request.remote_addr}")
    try:
        record_audio(AUDIO_FILENAME)
        transcribed_text = transcribe_audio(AUDIO_FILENAME)
        summary = summarize_text(transcribed_text)
        data = save_to_csv(transcribed_text, summary)
        # if os.path.exists(AUDIO_FILENAME):
        #     os.remove(AUDIO_FILENAME)
        #     logging.info(f"Deleted audio file: {AUDIO_FILENAME}")
        return jsonify(data)
    except Exception as e:
        logging.error(f"Record patient error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/record_chat', methods=['POST'], strict_slashes=False)
def record_chat():
    logging.info(f"Received POST request to /record_chat from {request.remote_addr}")
    try:
        record_audio(CHAT_AUDIO_FILENAME, duration=7)
        user_question = transcribe_audio(CHAT_AUDIO_FILENAME)
        df = pd.read_csv(CSV_FILE)
        system_prompt = "You are a helpful AI medical assistant."
        response = generate_response(system_prompt, user_question, df)
        # if os.path.exists(CHAT_AUDIO_FILENAME):
        #     os.remove(CHAT_AUDIO_FILENAME)
        #     logging.info(f"Deleted audio file: {CHAT_AUDIO_FILENAME}")
        return jsonify({"question": user_question, "answer": response})
    except Exception as e:
        logging.error(f"Record chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat_text', methods=['POST'], strict_slashes=False)
def chat_text():
    logging.info(f"Received POST request to /chat_text from {request.remote_addr}")
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            logging.warning("No message provided in /chat_text request")
            return jsonify({"error": "No message provided"}), 400
        df = pd.read_csv(CSV_FILE)
        system_prompt = "You are a helpful AI medical assistant."
        response = generate_response(system_prompt, user_input, df)
        return jsonify({"answer": response})
    except Exception as e:
        logging.error(f"Chat text error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug_audio', methods=['POST'], strict_slashes=False)
def debug_audio():
    logging.info(f"Received POST request to /debug_audio from {request.remote_addr}")
    try:
        if 'audio' not in request.files:
            logging.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files['audio']
        filename = f"debug_{uuid.uuid4().hex}.wav"
        audio_file.save(filename)
        file_size = os.path.getsize(filename)
        logging.info(f"Saved debug audio to {filename} (size: {file_size} bytes)")
        transcribed_text = transcribe_audio(filename)
        # os.remove(filename)
        return jsonify({"transcription": transcribed_text, "filename": filename})
    except Exception as e:
        logging.error(f"Debug audio error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'], strict_slashes=False)
def test():
    logging.info(f"Received POST request to /test from {request.remote_addr}")
    return jsonify({"status": "ok"})

@app.route('/routes')
def list_routes():
    logging.info("Listing all routes")
    return jsonify([str(rule) for rule in app.url_map.iter_rules()])

if __name__ == '__main__':
    logging.info("Starting Flask server on port 5001")
    app.run(debug=True, port=5001)

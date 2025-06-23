# 🏥 D-BOT: Doctor’s Bot for Operational Trackers 🤖  

**D-BOT** is an **AI-powered medical assistant** designed to help doctors and healthcare professionals **record, analyze, and retrieve patient data effortlessly**. It integrates **speech-to-text**, **AI-driven medical summarization**, and a **smart chatbot** to provide real-time assistance for medical professionals.  

---

## 📌 Features  

✅ **🎤 Voice-to-Text Patient Recording** – AI extracts key medical details from speech  
✅ **📄 AI-Powered Summarization** – Automatically summarizes patient name, age, symptoms, diagnosis, etc.  
✅ **📊 CSV Data Storage** – Securely stores patient records for future use  
✅ **💬 AI Chatbot Mode** – Retrieve patient data & get insights using AI  
✅ **🌙 Dark Mode UI** – User-friendly Bootstrap-based interface  

---

## 🚀 Installation  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/fardeenKhadri/D-BOTT.git
cd D-BOTT
```

### **2️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **3️⃣ Set Up API Keys**  
- Get API Keys from:  
  - [Google Gemini AI](https://ai.google.dev/)  
  - [Groq Whisper & Llama](https://groq.com/)  

- Open **`app.py`** and replace the placeholders with your API keys:  
```python
GROQ_API_KEY = "your_groq_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
```

---

## 🎬 Running D-BOT  

Start the Flask server:  
```sh
python app.py
```

Then, open your browser and visit:  
```
http://127.0.0.1:5000
```

---

## 🖥 Web Interface  

1️⃣ **Click "Add Patient Data"** → Speak, and AI will extract patient details.  
2️⃣ **Click "Ask AI"** → Ask D-BOT questions about stored patient records.  
3️⃣ **Type & Chat** → Directly type a query and get instant medical insights.  

---

## 📂 Project Structure  

```
📁 D-BOTT
│── app.py             # Flask Backend
│── templates/
│   ├── index.html     # Web UI
│── static/
│   ├── style.css      # Styling
│── patient_data.csv   # Stores patient history
│── requirements.txt   # Python Dependencies
│── README.md          # Documentation
```

---

## 🛠 Technologies Used  

- **Flask** – Web framework  
- **pyaudio & wave** – Audio recording  
- **Google Gemini AI** – Medical text summarization  
- **Groq Whisper** – Speech-to-text AI  
- **Groq Llama 3** – AI Chatbot for medical insights  
- **Bootstrap & JavaScript** – Frontend UI  

---

## 📞 Support  

For issues, create a **GitHub issue** or contact **skkstew@gmail.com**.  

🚀 **D-BOT is here to assist doctors in tracking patient records smarter and faster!** 🏥🎤🤖  

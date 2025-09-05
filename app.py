from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import fitz  # pymupdf
from dotenv import load_dotenv
from flask_cors import CORS
import uuid
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load API Key Gemini
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in environment variables.")

genai.configure(api_key=api_key)

# Fungsi baca CV (PDF)
def read_cv_text(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            print(f"CV file not found: {pdf_path}")
            return ""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Failed to read CV: {e}")
        return ""
    
# fungsi baca json
def read_json_file(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return {}

# Ambil isi CV untuk dijadikan konteks tambahan
cv_text = read_cv_text("data/cv.pdf")
projects_data = read_json_file("data/projects.json")

# Instruksi sistem (tetap singkat, tidak dicampur data CV langsung)
instruction = (
    "Kamu adalah chatbot pribadi dari Irsyan Ramadhan, seorang mahasiswa Teknik Komputer "
    "yang memiliki minat dalam web development, mobile development, dan deep learning. "
    "Jawab pertanyaan tentang Irsyan berdasarkan data CV dan daftar proyek yang sudah diberikan. "
    "Jika tidak ada informasi yang relevan, katakan: 'Silakan bertanya langsung kepada Irsyan melalui form kontak.' "
    "Ikuti bahasa yang digunakan pengunjung."
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=instruction
)

# Sesi per user
chat_sessions = {}

# ========== Anti Prompt Injection Filter ==========
def is_prompt_injection(message: str) -> bool:
    suspicious_patterns = [
        # English
        "ignore previous", "disregard", "forget instructions",
        "system prompt", "reveal", "api key", "override", "jailbreak",
        "remove safety", "disable filter", "bypass",
        # Indonesian
        "abaikan instruksi", "lupakan perintah", "hapus aturan",
        "tunjukkan api key", "bocorkan", "abaikan aturan",
        "matikan filter", "lewati aturan"
    ]
    lower_msg = message.lower()
    return any(p in lower_msg for p in suspicious_patterns)

# =================================================

@app.route('/')
def health_check():
    return jsonify({'status': 'OK', 'message': 'Irsyan Ramadhan Portfolio Chatbot is running!'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # 🔒 Cek apakah ada indikasi prompt injection
    if is_prompt_injection(user_message):
        return jsonify({
            'reply': "Permintaan ini tidak valid. Silakan hanya bertanya tentang pengalaman, proyek, skill, atau pendidikan Irsyan.",
            'session_id': session_id or "invalid"
        })

    # Buat session_id jika belum ada
    if not session_id:
        session_id = str(uuid.uuid4())

    # Ambil atau buat sesi
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat()

    try:
        # Tambahkan konteks CV & Project setiap kali user bertanya
        context_message = (
            f"CV data:\n{cv_text}\n\n"
            f"Projects data:\n{json.dumps(projects_data, indent=2)}\n\n"
            f"User question: {user_message}"
        )

        response = chat_sessions[session_id].send_message(context_message)
        return jsonify({'reply': response.text.strip(), 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': f'AI error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

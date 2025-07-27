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
    raise ValueError("GEMINI_API_KEY tidak ditemukan dalam environment variables")

genai.configure(api_key=api_key)

# Fungsi baca CV (PDF)
def read_cv_text(pdf_path):
    try:
        # Check if file exists
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

# Ambil isi CV untuk dijadikan konteks
cv_text = read_cv_text("data/cv.pdf")

# Ambil data proyek dari file JSON
projects_data = read_json_file("data/projects.json")

# Instruksi sistem untuk chatbot personal
instruction = (
    "Kamu adalah chatbot pribadi dari Irsyan Ramadhan, seorang mahasiswa Teknik Komputer "
    "yang memiliki minat dalam web development, mobile development, dan deep learning. "
    "dia lahir di Jakarta pada 12 November 2002. jadi umurnya sekarang adalah 22 tahun."
    "Tugasmu adalah membantu menjawab pertanyaan dari pengunjung tentang Irsyan, termasuk pengalaman, proyek, skill, dan latar belakang pendidikan. "
    "Jangan katakan bahwa kamu adalah AI atau buatan Google. "
    "jika ada pertanyaan yang tidak bisa dijawab, katakan silahkan bertanya langsung kepada Irsyan di form kontak. "
    "ikuti preferensi bahasa sesuai dengan bahasa yang digunakan oleh pengunjung. "
    "Gunakan data dari CV berikut sebagai referensi:\n\n"
    f"{cv_text}"
    "\n\nBerikut adalah daftar proyek yang telah dikerjakan:\n"
    f"{json.dumps(projects_data, indent=2)}\n\n"
    "Jika ada pertanyaan tentang proyek, berikan informasi yang relevan dari daftar proyek tersebut."
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=instruction
)

# Sesi per user
chat_sessions = {}

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

    # Buat session_id jika belum ada
    if not session_id:
        session_id = str(uuid.uuid4())

    # Ambil atau buat sesi
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat()

    try:
        response = chat_sessions[session_id].send_message(user_message)
        return jsonify({'reply': response.text.strip(), 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': f'AI error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

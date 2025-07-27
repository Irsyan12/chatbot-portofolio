from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)


# Load data produk dari CSV
produk_df = pd.read_csv("produk.csv")

# Konfigurasi Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan dalam environment variables")
genai.configure(api_key=api_key)

# Fungsi bantu untuk menyusun daftar produk sebagai konteks
def get_produk_context():
    rows = []
    for _, row in produk_df.iterrows():
        rows.append(f"- {row['nama_produk']} ({row['kategori']}), harga: Rp{row['harga']}, stok: {row['stok']}")
    return "\n".join(rows)

# Inisialisasi model Gemini dan session chat
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "Kamu adalah customer service bernama Tessa dari toko Texas Market. "
        "Jangan pernah menyebutkan kamu adalah AI atau dari Google. "
        "Tugasmu adalah melayani pelanggan dengan ramah dan profesional. "
        "Jika pelanggan bertanya tentang stok atau ingin memesan, gunakan daftar produk yang tersedia untuk menjawab."
        "jika pelanggan bertanya tentang produk, berikan informasi yang relevan. "
        "jika pelanggan sudah ingin memesan, tanyakan nama dan alamat pengiriman dengan format yang jelas. "
        "jika pelanggan sudah mengisikan alamat, minta pelanggan untuk mentransfer uang sesuai total belanja ke rekening BSI 7195304698 a/n Irsyan Ramadhan."
        "Berikut daftar produk:\n" + get_produk_context()
    )
)

# Simpan sesi chat ke dalam app (bersifat global untuk saat ini)
chat_session = model.start_chat()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Kirim pesan ke Gemini dengan chat session (agar ingat percakapan)
        response = chat_session.send_message(user_message)
        ai_reply = response.text.strip()

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Log error untuk debugging
        return jsonify({'error': f'AI service error: {str(e)}'}), 500

    return jsonify({'reply': ai_reply})

if __name__ == '__main__':
    app.run(debug=True)

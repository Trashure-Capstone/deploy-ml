# Base image menggunakan Python versi terbaru
FROM python:3.9-slim

# Set working directory ke /app
WORKDIR /app

# Install dependensi yang diperlukan
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh konten proyek ke direktori /app
COPY . .

# Jalankan aplikasi dengan Gunicorn
CMD gunicorn -b :$PORT main:app

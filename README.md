# Deteksi Jerawat Berbasis AI

Proyek ini adalah aplikasi berbasis **Python + AI (YOLO/CNN)** untuk mendeteksi jenis-jenis jerawat pada wajah secara otomatis melalui gambar atau kamera. Aplikasi ini dirancang untuk membantu pengguna mengenali kondisi kulit mereka dengan mudah dan cepat.

## 📌 Fitur Utama

- 🔍 Deteksi otomatis jerawat menggunakan model AI (YOLOv8 / CNN)
- 📷 Input dari kamera atau gambar statis
- 🖼 Tampilan antarmuka sederhana
- 📁 Folder dataset dan hasil deteksi
- 💡 Cocok untuk edukasi, demo, atau riset AI Computer Vision

## 🛠 Teknologi yang Digunakan

- Python 3
- OpenCV
- YOLOv8 (Ultralytics)
- Tkinter (jika ada GUI)
- NumPy, matplotlib, dll

## 📂 Struktur Folder

```bash
.
├── dataset/              # Dataset gambar jerawat
├── models/               # Model YOLO terlatih
├── main.py               # Script utama untuk deteksi
├── requirements.txt      # Daftar dependensi
└── README.md             # Dokumentasi proyek ini

git clone https://github.com/Asepteknik98/deteksi_jerawat.git
cd deteksi_jerawat

pip install -r requirements.txt

python main.py

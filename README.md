# 💧 Fossil - Aplikasi Prediksi Kelayakan Air Minum  
_Tugas Besar GDGoC Telkom University_

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://fossil-app.streamlit.app/)

## 🧠 Deskripsi

**Fossil** adalah aplikasi berbasis Machine Learning yang dirancang untuk memprediksi apakah air layak dikonsumsi (potable) atau tidak. Aplikasi ini bekerja dengan menganalisis parameter-parameter kimia dan fisik dari sampel air, seperti **pH**, **kadar sulfat**, **jumlah karbon organik**, dan lainnya.

Aplikasi ini dikembangkan sebagai bagian dari **Tugas Besar GDGoC (Google Developer Student Clubs Open Class)** di **Telkom University**.

## 🔍 Cara Kerja Model

- **Model**: Neural Network (`MLPClassifier`)
- **Hidden Layers**: (100, 50)
- **Aktivasi**: ReLU
- **Optimizer**: Adam
- **Epoch Maksimum**: 500 iterasi
- **Dataset**: [Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Input**: 9 parameter kualitas air
- **Output**: Prediksi "Potable" atau "Not Potable"

## 🌐 Coba Aplikasi

Akses aplikasi web-nya di sini:  
🔗 **[https://fossil-app.streamlit.app/](https://fossil-app.streamlit.app/)**

## 📁 Struktur Repositori

```
.
├── .devcontainer/            # Konfigurasi untuk dev container (opsional)
├── FOSSIL_GDGoC.ipynb        # Notebook eksplorasi awal & pelatihan model
├── gb_model.pkl              # File model terlatih (jika pakai Gradient Boosting)
├── main.py                   # Aplikasi Streamlit utama
├── requirements.txt          # Daftar dependensi
├── water_potability.csv      # Dataset utama
└── README.md                 # Dokumentasi proyek
```

## ⚙️ Cara Menjalankan Lokal

1. Clone repositori:
   ```bash
   git clone https://github.com/Fauzan-A25/FOSSIL_GDGoC-TUBES.git
   cd FOSSIL_GDGoC-TUBES
   ```

2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run main.py
   ```

## 👨‍💻 Kelompok Pengembang Aplikasi

- **Fauzan Ahsanudin Alfikri** – 103052300003  
- **Benedict Joel Purba** – 103052300066  

> Aplikasi ini dikembangkan sebagai bagian dari **Tugas Besar GDGoC** di **Telkom University**.

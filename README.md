# Penelitian Pengaruh Penggunaan Media Sosial terhadap Kesehatan Mental

## Deskripsi Proyek

Proyek ini bertujuan untuk mengeksplorasi hubungan antara kebiasaan penggunaan media sosial dan kondisi kesehatan mental individu. Dengan menggunakan teknik pembelajaran mesin (Machine Learning), penelitian ini menerapkan pendekatan **unsupervised** dan **supervised learning** untuk menggali pola perilaku digital yang memengaruhi kesejahteraan psikologis.

### Pendekatan yang Digunakan

#### 1. **Unsupervised Learning: K-Means Clustering**
Pada pendekatan unsupervised, kami menggunakan **K-Means Clustering** untuk mengelompokkan individu berdasarkan kemiripan respons mereka terhadap berbagai faktor seperti:
- Intensitas penggunaan media sosial
- Gangguan konsentrasi
- Pencarian validasi sosial
- Tingkat kecemasan

Proses ini bertujuan untuk mengidentifikasi segmen-segmen pengguna yang memiliki karakteristik perilaku dan emosional serupa, yang nantinya dapat digunakan untuk analisis lebih lanjut.

#### 2. **Supervised Learning: Random Forest**
Untuk pendekatan supervised, kami membangun model prediktif menggunakan algoritma **Random Forest**. Model ini digunakan untuk memprediksi kemungkinan seseorang mengalami:
- Depresi
- Gangguan tidur

Prediksi ini didasarkan pada variabel-variabel seperti:
- Usia
- Jenis kelamin
- Status hubungan
- Intensitas dan motivasi penggunaan media sosial

Model ini diharapkan memberikan hasil yang akurat dalam mendeteksi risiko kesehatan mental secara dini.

## Tujuan dan Manfaat

Penelitian ini tidak hanya bertujuan untuk memahami lebih dalam pola perilaku digital yang memengaruhi kondisi psikologis seseorang, tetapi juga berkontribusi dalam pengembangan solusi berbasis data untuk mendukung kesehatan mental di era digital. Dengan pendekatan berbasis teknologi, kami berharap dapat menciptakan solusi yang lebih efektif dan terukur dalam menangani isu-isu kesehatan mental yang semakin relevan di dunia digital saat ini.

## Fitur Utama

- **K-Means Clustering** untuk mengelompokkan individu berdasarkan pola penggunaan media sosial dan kondisi psikologis.
- **Random Forest** untuk memprediksi kondisi kesehatan mental, seperti depresi dan gangguan tidur, berdasarkan data demografis dan perilaku digital.
- Visualisasi hasil yang jelas dan mudah dipahami untuk mendukung interpretasi data.

## Teknologi yang Digunakan

- **Python** (untuk pemrosesan data dan machine learning)
- **Pandas** dan **Scikit-Learn** (untuk analisis data dan pembelajaran mesin)
- **Matplotlib** dan **Seaborn** (untuk visualisasi data)

## Cara Menggunakan

1. Clone repository ini ke mesin lokal Anda:
   ```bash
   git clone https://github.com/username/repo-name.git

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA

# --- 2. Konfigurasi Halaman & Fungsi Helper ---
st.set_page_config(page_title="Dashboard Analisis Mental Health", layout="wide")

@st.cache_data
def load_data(file_path='smmh.csv'):
    try:
        df = pd.read_csv(file_path)
        # Membersihkan dan me-rename nama kolom
        df.columns = df.columns.str.replace(r'^\d+\W*\s*|\s*\(\d+-\d+\)$|\?$|^\s*|\s*$', '', regex=True)
        df.rename(columns={
            'What is your age': 'Age', 'Gender': 'Gender', 'Relationship Status': 'Relationship_Status',
            'Occupation Status': 'Occupation_Status', 'Do you use social media': 'Use_Social_Media',
            'What is the average time you spend on social media every day': 'Avg_Time_Social_Media',
            'How often do you find yourself using Social media without a specific purpose': 'SM_No_Purpose',
            'How often do you get distracted by Social media when you are busy doing something': 'SM_Distraction_Busy',
            'Do you feel restless if you haven_t used Social media in a while': 'SM_Restless',
            'On a scale of 1 to 5, how easily distracted are you': 'Easily_Distracted',
            'On a scale of 1 to 5, how much are you bothered by worries': 'Bothered_By_Worries',
            'Do you find it difficult to concentrate on things': 'Difficult_To_Concentrate',
            'On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media': 'SM_Compare_Success',
            'Following the previous question, how do you feel about these comparisons, generally': 'SM_Compare_Feeling',
            'How often do you look to seek validation from features of social media': 'SM_Seek_Validation',
            'How often do you feel depressed or down': 'Feel_Depressed',
            'On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate': 'Interest_Fluctuation',
            'On a scale of 1 to 5, how often do you face issues regarding sleep': 'Sleep_Issues'
        }, inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"🚨 File '{file_path}' tidak ditemukan!")
        return None

# --- Judul Utama Aplikasi ---
st.title("📊 Dashboard Analisis: Media Sosial & Kesehatan Mental")
st.markdown("Sebuah dashboard interaktif yang mengikuti kerangka **CRISP-DM** untuk menganalisis data survei.")

# --- Load Data di Awal ---
df = load_data()

if df is None:
    st.stop()

# ==============================================================================
# TAHAP 1: DATA UNDERSTANDING
# ==============================================================================
st.header("🎯 Tahap 1: Data Understanding")
st.markdown("Mengumpulkan, mendeskripsikan, dan melakukan eksplorasi awal untuk mengenali data yang relevan.")

with st.expander("Lihat Detail Tahap Data Understanding"):
    st.subheader("Deskripsi Data Awal")
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Responden", f"{df.shape[0]:,}")
    col2.metric("Jumlah Variabel", f"{df.shape[1]}")
    st.dataframe(df.head())
    
    st.subheader("Distribusi Demografis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['Age'], kde=True, bins=20, ax=axes[0])
    axes[0].set_title('Distribusi Usia')
    if 'Gender' in df.columns:
        df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1], startangle=90)
        axes[1].set_title('Distribusi Gender')
        axes[1].set_ylabel('')
    st.pyplot(fig)

    st.subheader("Analisis Korelasi dengan Heatmap")
    st.markdown("Heatmap ini membantu kita melihat hubungan antar variabel numerik. Semakin mendekati 1 (merah) atau -1 (biru), semakin kuat hubungannya.")
    
    # Persiapan data untuk korelasi
    kolom_numerik_corr = ['Age', 'SM_No_Purpose', 'SM_Distraction_Busy', 'Easily_Distracted', 'Bothered_By_Worries', 'Difficult_To_Concentrate', 'SM_Compare_Success', 'SM_Seek_Validation', 'Feel_Depressed', 'Interest_Fluctuation', 'Sleep_Issues']
    kolom_ordinal_corr = ['Avg_Time_Social_Media']
    kolom_valid_numerik = [col for col in kolom_numerik_corr if col in df.columns]
    kolom_valid_ordinal = [col for col in kolom_ordinal_corr if col in df.columns]
    df_corr = df[kolom_valid_numerik + kolom_valid_ordinal].copy()
    
    time_categories_corrected = ['Less than an Hour', 'Between 1 and 2 hours', 'Between 2 and 3 hours', 'Between 3 and 4 hours', 'Between 4 and 5 hours', 'More than 5 hours']
    ordinal_encoder = OrdinalEncoder(categories=[time_categories_corrected])
    df_corr['Avg_Time_Social_Media_Encoded'] = ordinal_encoder.fit_transform(df_corr[['Avg_Time_Social_Media']])
    df_corr_final = df_corr.drop(columns=['Avg_Time_Social_Media'])
    
    correlation_matrix = df_corr_final.corr()
    
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, ax=ax)
    st.pyplot(fig)

# ==============================================================================
# TAHAP 2: DATA PREPARATION
# ==============================================================================
st.header("🛠️ Tahap 2: Data Preparation")
st.markdown("Membersihkan dan menyiapkan data untuk diolah oleh model analitik.")

# --- Sidebar untuk Input Kontrol ---
st.sidebar.header("⚙️ Pengaturan Model")
k_optimal_input = st.sidebar.number_input("Pilih Jumlah Cluster (K)", 2, 10, 3, 1)
depression_threshold = st.sidebar.slider("Threshold 'Depresi' (Skor >= Threshold → Ya)", 1, 5, 4, 1)
sleep_threshold = st.sidebar.slider("Threshold 'Gangguan Tidur'", 1, 5, 4, 1)

# Lakukan feature engineering & binarisasi di sini
df['KMeans_Gangguan_Konsentrasi'] = df[['SM_Distraction_Busy', 'Easily_Distracted', 'Difficult_To_Concentrate']].mean(axis=1)
df['KMeans_Pencarian_Validasi'] = df[['SM_Compare_Success', 'SM_Seek_Validation']].mean(axis=1)
df['Target_Depresi'] = df['Feel_Depressed'].apply(lambda x: 1 if x >= depression_threshold else 0)
df['Target_Gangguan_Tidur'] = df['Sleep_Issues'].apply(lambda x: 1 if x >= sleep_threshold else 0)

with st.expander("Lihat Detail Tahap Data Preparation"):
    st.markdown("""
    1.  **Feature Engineering**: Membuat fitur `KMeans_Gangguan_Konsentrasi` dan `KMeans_Pencarian_Validasi` dengan merata-ratakan beberapa pertanyaan terkait untuk mendapatkan skor yang lebih representatif.
    2.  **Binarisasi Target**: Mengubah variabel target (`Feel_Depressed`, `Sleep_Issues`) menjadi biner (0 atau 1) berdasarkan threshold yang dipilih di sidebar. Ini diperlukan untuk model klasifikasi.
    3.  **Encoding & Scaling**: Proses ini akan dilakukan di dalam pipeline model untuk memastikan konsistensi antara data latih dan data baru.
    """)
    st.dataframe(df[['KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Target_Depresi', 'Target_Gangguan_Tidur']].head())

# ==============================================================================
# TAHAP 3: MODELING
# ==============================================================================
st.header("📈 Tahap 3: Modeling")
st.markdown("Membangun model analitik untuk menemukan pola (K-Means) dan membuat prediksi (Naïve Bayes).")

with st.expander("Lihat Detail Tahap Modeling & Evaluasi"):
    st.subheader("3.1 Modeling Unsupervised: K-Means Clustering")
    # Preprocessing untuk K-Means
    kolom_kmeans_numerik = ['SM_No_Purpose', 'KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Bothered_By_Worries']
    df_kmeans_selection = df[kolom_kmeans_numerik + ['Avg_Time_Social_Media']].copy()
    ordinal_encoder_kmeans = OrdinalEncoder(categories=[time_categories_corrected])
    df_kmeans_selection['Avg_Time_Social_Media_Encoded'] = ordinal_encoder_kmeans.fit_transform(df_kmeans_selection[['Avg_Time_Social_Media']])
    df_kmeans_processed = df_kmeans_selection.drop(columns=['Avg_Time_Social_Media'])
    kolom_kmeans_final = ['Avg_Time_Social_Media_Encoded'] + kolom_kmeans_numerik
    df_kmeans_processed = df_kmeans_processed[kolom_kmeans_final]
    for col in kolom_kmeans_final:
        if df_kmeans_processed[col].isnull().any():
            df_kmeans_processed[col] = df_kmeans_processed[col].fillna(df_kmeans_processed[col].median())
    scaler_kmeans = StandardScaler()
    kmeans_scaled = scaler_kmeans.fit_transform(df_kmeans_processed)

    # Menjalankan K-Means dan menampilkan hasilnya
    kmeans = KMeans(n_clusters=k_optimal_input, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(kmeans_scaled)
    centroids_df = pd.DataFrame(scaler_kmeans.inverse_transform(kmeans.cluster_centers_), columns=kolom_kmeans_final)
    
    st.markdown("##### Karakteristik Rata-Rata per Cluster (Centroids)")
    st.dataframe(centroids_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='pink'))

    st.subheader("3.2 Modeling Supervised: Naïve Bayes Classifier")
    st.markdown("Model Naïve Bayes dibangun untuk memprediksi **Indikasi Depresi** dan **Indikasi Gangguan Tidur**.")
    st.info("Hasil evaluasi detail dari model ini akan ditampilkan di tahap **Evaluation**.")

# ==============================================================================
# TAHAP 4: EVALUATION
# ==============================================================================
st.header("✅ Tahap 4: Evaluation")
st.markdown("Mengevaluasi performa dan ketepatan model untuk memastikan model memenuhi tujuan.")

with st.expander("Lihat Detail Tahap Evaluation"):
    # Mendefinisikan fitur dan preprocessor untuk Naive Bayes
    fitur_nb_numerik = ['Age', 'SM_No_Purpose']
    fitur_nb_kategori_nominal = ['Gender', 'Relationship_Status']
    fitur_nb_kategori_ordinal = ['Avg_Time_Social_Media']
    kolom_fitur_nb_gabungan = fitur_nb_numerik + fitur_nb_kategori_nominal + fitur_nb_kategori_ordinal
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=[time_categories_corrected]), fitur_nb_kategori_ordinal),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), fitur_nb_kategori_nominal),
            ('num', StandardScaler(), fitur_nb_numerik)],
        remainder='drop')
    
    X = df[kolom_fitur_nb_gabungan]
    targets_to_eval = {"Indikasi Depresi": df['Target_Depresi'], "Indikasi Gangguan Tidur": df['Target_Gangguan_Tidur']}

    for name, y in targets_to_eval.items():
        st.subheader(f"Hasil Evaluasi untuk: {name}")
        if y.nunique() < 2: st.warning("Target hanya punya satu kelas."); continue
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())])
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        # Menampilkan metrik dalam kolom
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Akurasi Model", f"{accuracy_score(y_test, y_pred):.2%}")
            st.metric("Skor ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
        with col2:
            st.text("Laporan Klasifikasi:")
            st.text(classification_report(y_test, y_pred, zero_division=0))
        
        # Menampilkan plot dalam kolom
        fig_eval, axes_eval = plt.subplots(1, 2, figsize=(12, 5))
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes_eval[0], cmap='Blues')
        axes_eval[0].set_title("Confusion Matrix")
        axes_eval[0].set_xlabel('Prediksi'); axes_eval[0].set_ylabel('Aktual')
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes_eval[1].plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
        axes_eval[1].plot([0, 1], [0, 1], linestyle='--', label='Garis Acak')
        axes_eval[1].set_title("Kurva ROC")
        axes_eval[1].set_xlabel('False Positive Rate'); axes_eval[1].set_ylabel('True Positive Rate')
        axes_eval[1].legend()
        st.pyplot(fig_eval)

        # Analisis Variabel Penting (Proxy untuk Naive Bayes)
        st.markdown("**Analisis Variabel Penting (Proxy)**")
        st.info("Karena Naïve Bayes tidak memiliki `feature_importances_` seperti Random Forest, kita melihat 'kepentingan' variabel dengan cara membandingkan distribusinya terhadap kelas target.")
        
        df_eval = X_test.copy()
        df_eval['Target'] = y_test
        
        # Visualisasi distribusi fitur numerik terhadap target
        fig_feat, axes_feat = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(x='Target', y='Age', data=df_eval, ax=axes_feat[0])
        axes_feat[0].set_title('Distribusi Usia terhadap Target')
        sns.boxplot(x='Target', y='SM_No_Purpose', data=df_eval, ax=axes_feat[1])
        axes_feat[1].set_title('Distribusi "Medsos Tanpa Tujuan" terhadap Target')
        st.pyplot(fig_feat)
        st.markdown("Dari plot di atas, kita bisa melihat apakah ada perbedaan distribusi yang signifikan antara kelas 0 (Tidak Berisiko) dan kelas 1 (Berisiko). Jika distribusinya sangat berbeda, artinya fitur tersebut penting untuk membedakan kelas.")
        st.markdown("---")


# ==============================================================================
# TAHAP 5: DEPLOYMENT
# ==============================================================================
st.header("🚀 Tahap 5: Deployment")
st.markdown("""
Tahap akhir dari CRISP-DM adalah _deployment_, di mana hasil dari analisis dan model diterapkan untuk digunakan oleh pengguna. **Dashboard interaktif ini adalah bentuk dari deployment tersebut.**

Di bawah ini adalah fitur prediksi interaktif yang menggunakan model **Naïve Bayes** yang telah dilatih untuk memberikan prediksi secara _real-time_ berdasarkan input dari pengguna.
""")

with st.form("prediction_form"):
    st.subheader("🔮 Coba Prediksi Sendiri")
    col1, col2 = st.columns(2)
    age = col1.number_input("Usia Anda:", 10, 100, 25, 1)
    gender = col2.selectbox("Jenis Kelamin:", df['Gender'].unique())
    relationship = col1.selectbox("Status Hubungan:", df['Relationship_Status'].unique())
    time = col2.selectbox("Rata-rata waktu di medsos per hari:", time_categories_corrected)
    no_purpose = st.slider("Seberapa sering menggunakan medsos tanpa tujuan? (1-5)", 1, 5, 3)
    submit = st.form_submit_button("Dapatkan Prediksi")

if submit:
    input_df = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'Relationship_Status': [relationship],
        'Avg_Time_Social_Media': [time], 'SM_No_Purpose': [no_purpose]
    }, columns=kolom_fitur_nb_gabungan)
    
    st.subheader("Hasil Prediksi Anda:")
    
    # Melatih ulang model dengan data lengkap untuk prediksi final
    X_full = df[kolom_fitur_nb_gabungan]
    y_depresi_full = df['Target_Depresi']
    y_tidur_full = df['Target_Gangguan_Tidur']
    
    # Prediksi Depresi
    model_depresi = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())]); model_depresi.fit(X_full, y_depresi_full)
    pred_d = model_depresi.predict(input_df)[0]; proba_d = model_depresi.predict_proba(input_df)[0]
    hasil_d = "Berisiko Tinggi 😢" if pred_d == 1 else "Berisiko Rendah 😊"
    st.write(f"**Indikasi Depresi:** {hasil_d}")
    st.progress(float(proba_d[1]), text=f"Probabilitas Berisiko Tinggi: {proba_d[1]:.0%}")

    # Prediksi Gangguan Tidur
    model_tidur = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GaussianNB())]); model_tidur.fit(X_full, y_tidur_full)
    pred_t = model_tidur.predict(input_df)[0]; proba_t = model_tidur.predict_proba(input_df)[0]
    hasil_t = "Berisiko Tinggi 😴" if pred_t == 1 else "Berisiko Rendah 🛌"
    st.write(f"**Indikasi Gangguan Tidur:** {hasil_t}")
    st.progress(float(proba_t[1]), text=f"Probabilitas Berisiko Tinggi: {proba_t[1]:.0%}")

    with st.expander("Bagaimana Cara Kerja Prediksi Naïve Bayes? (Penjelasan Peluang)"):
        st.markdown("""
        Model Naïve Bayes bekerja berdasarkan **Teorema Bayes**. Secara sederhana, untuk memprediksi apakah Anda 'Berisiko' atau 'Tidak', model menghitung dua probabilitas:
        1.  P(Berisiko | Input Anda) : Peluang Anda **berisiko**, *mengingat* data input (usia, gender, dll.) yang Anda berikan.
        2.  P(Tidak Berisiko | Input Anda) : Peluang Anda **tidak berisiko**, *mengingat* data input yang Anda berikan.
        
        Model kemudian akan memilih kelas dengan probabilitas tertinggi. Probabilitas ini dihitung dengan mengalikan probabilitas dari masing-masing fitur Anda. Contohnya (sangat disederhanakan):
        
        *P(Berisiko | Input) ~ P(Usia=25 | Berisiko) x P(Gender=Pria | Berisiko) x P(Waktu=Lama | Berisiko) x P(Berisiko)*
        
        Setiap komponen probabilitas ini (seperti *P(Usia=25 | Berisiko)*) "dipelajari" oleh model dari data latih yang sudah ada.
        """)
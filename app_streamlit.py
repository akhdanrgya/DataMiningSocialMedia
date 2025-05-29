# app_streamlit_local_data.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Analisis Kesehatan Mental & Media Sosial", layout="wide")

# --- Fungsi-Fungsi Pembantu ---

@st.cache_data # Cache data loading and basic preprocessing
def load_and_preprocess_data(file_path='smmh.csv'): # File path default ke 'smmh.csv'
    try:
        df = pd.read_csv(file_path)
        # Mengganti nama kolom agar lebih mudah dipanggil
        df.columns = df.columns.str.replace(r'^\d+\W*\s*|\s*\(\d+-\d+\)$|\?$|^\s*|\s*$', '', regex=True)
        df.rename(columns={
            'What is your age': 'Age',
            'Gender': 'Gender',
            'Relationship Status': 'Relationship_Status',
            'Occupation Status': 'Occupation_Status',
            'Do you use social media': 'Use_Social_Media',
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
        
        # Membuat fitur komposit untuk K-Means
        df['KMeans_Gangguan_Konsentrasi'] = df[['SM_Distraction_Busy', 'Easily_Distracted', 'Difficult_To_Concentrate']].mean(axis=1)
        df['KMeans_Pencarian_Validasi'] = df[['SM_Seek_Validation', 'SM_Compare_Success']].mean(axis=1)
        
        return df
    except FileNotFoundError:
        st.error(f"🚨 File '{file_path}' tidak ditemukan! Pastikan file tersebut ada di direktori yang sama dengan skrip Streamlit ini.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat atau memproses data dari '{file_path}': {e}")
        return None

# (Fungsi preprocess_for_kmeans dan preprocess_for_rf tetap sama seperti sebelumnya, jadi tidak gue tulis ulang di sini biar ringkas)
# Salin fungsi preprocess_for_kmeans dan preprocess_for_rf dari kode Streamlit sebelumnya ke sini

def preprocess_for_kmeans(df_input):
    try:
        kolom_kmeans_numerik = ['SM_No_Purpose', 'KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Bothered_By_Worries']
        kolom_kmeans_kategori_ordinal = ['Avg_Time_Social_Media']
        
        df_kmeans_selection = df_input[kolom_kmeans_numerik + kolom_kmeans_kategori_ordinal].copy()

        time_categories = ['Less than 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', 'More than 5 hours']
        ordinal_encoder_time = OrdinalEncoder(categories=[time_categories], handle_unknown='use_encoded_value', unknown_value=np.nan) 
        
        df_kmeans_selection['Avg_Time_Social_Media_Encoded'] = ordinal_encoder_time.fit_transform(df_kmeans_selection[['Avg_Time_Social_Media']])
        df_kmeans_processed = df_kmeans_selection.drop(columns=['Avg_Time_Social_Media'])
        kolom_kmeans_final = ['Avg_Time_Social_Media_Encoded'] + kolom_kmeans_numerik

        for col in kolom_kmeans_final:
            if df_kmeans_processed[col].isnull().any():
                df_kmeans_processed[col] = df_kmeans_processed[col].fillna(df_kmeans_processed[col].median())
        
        scaler_kmeans = StandardScaler()
        kmeans_scaled = scaler_kmeans.fit_transform(df_kmeans_processed[kolom_kmeans_final])
        return kmeans_scaled, kolom_kmeans_final, df_kmeans_processed
    except Exception as e:
        st.error(f"Error saat preprocessing untuk K-Means: {e}")
        return None, None, None

def preprocess_for_rf(df_input, depression_threshold, sleep_threshold):
    try:
        fitur_rf_numerik = ['Age', 'SM_No_Purpose'] 
        fitur_rf_kategori_nominal = ['Gender', 'Relationship_Status']
        fitur_rf_kategori_ordinal = ['Avg_Time_Social_Media']
        
        time_categories = ['Less than 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', 'More than 5 hours']

        preprocessor_rf = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(categories=[time_categories], handle_unknown='use_encoded_value', unknown_value=-1), fitur_rf_kategori_ordinal),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), fitur_rf_kategori_nominal),
                ('num', StandardScaler(), fitur_rf_numerik)
            ], 
            remainder='drop'
        )
        
        df_input['Target_Depresi'] = df_input['Feel_Depressed'].apply(lambda x: 1 if x >= depression_threshold else 0)
        df_input['Target_Gangguan_Tidur'] = df_input['Sleep_Issues'].apply(lambda x: 1 if x >= sleep_threshold else 0)

        kolom_fitur_rf_gabungan = fitur_rf_numerik + fitur_rf_kategori_nominal + fitur_rf_kategori_ordinal
        
        missing_cols_X = [col for col in kolom_fitur_rf_gabungan if col not in df_input.columns]
        if missing_cols_X:
            st.error(f"Kolom fitur RF hilang: {missing_cols_X}")
            return None, None, None, None, None
            
        X = df_input[kolom_fitur_rf_gabungan]
        y_depresi = df_input['Target_Depresi']
        y_tidur = df_input['Target_Gangguan_Tidur']
        
        return X, y_depresi, y_tidur, preprocessor_rf
    except Exception as e:
        st.error(f"Error saat preprocessing untuk Random Forest: {e}")
        # return None, None, None, None, None # This line was commented, ensure it's correct
        return None, None, None, None # Adjusted to return 4 Nones if preprocessor_rf is the last one expected

# --- Judul Aplikasi ---
st.title("📊 Dashboard Analisis Penggunaan Media Sosial dan Kesehatan Mental")
st.markdown("""
Aplikasi ini melakukan analisis data dari survei penggunaan media sosial dan kesehatan mental. 
Diasumsikan file `smmh.csv` sudah ada di direktori yang sama dengan aplikasi ini.
Analisis mencakup segmentasi pengguna dengan K-Means Clustering dan prediksi status kesehatan mental 
(indikasi depresi & gangguan tidur) menggunakan Random Forest.
""")

# --- Memuat Data Langsung ---
DATA_FILE_PATH = 'smmh.csv' # Nama file data yang diharapkan
df_raw = load_and_preprocess_data(DATA_FILE_PATH)

# --- Sidebar untuk Input (selain file upload) ---
st.sidebar.header("⚙️ Pengaturan Analisis")

if df_raw is None:
    # Pesan error sudah ditampilkan oleh load_and_preprocess_data
    st.warning("Analisis tidak dapat dilanjutkan karena data gagal dimuat.")
    st.stop() # Hentikan eksekusi jika data tidak bisa dimuat
else:
    st.success(f"🎉 Dataset '{DATA_FILE_PATH}' berhasil dimuat dan diproses awal!")
    
    if st.checkbox("Tampilkan Sampel Data Mentah (Setelah Rename)", False): # Opsi ini bisa dipindah ke main panel
        st.subheader("Sampel Data Awal")
        st.dataframe(df_raw.head())
        st.write(f"Jumlah baris: {df_raw.shape[0]}, Jumlah kolom: {df_raw.shape[1]}")

    # --- K-Means Clustering Section ---
    # (Kode untuk K-Means Clustering tetap sama seperti sebelumnya)
    # Salin bagian K-Means dari kode Streamlit sebelumnya ke sini
    st.header("🧩 1. K-Means Clustering: Segmentasi Pengguna")
    st.markdown("""
    Mengelompokkan responden berdasarkan pola kebiasaan penggunaan media sosial dan gejala kesehatan mental 
    (intensitas penggunaan, gangguan konsentrasi, pencarian validasi, tingkat kecemasan).
    """)

    kmeans_scaled_data, kmeans_feature_names, df_kmeans_original_features = preprocess_for_kmeans(df_raw.copy())

    if kmeans_scaled_data is not None:
        st.subheader("Menentukan Jumlah Cluster (K) Optimal - Elbow Method")
        inertia = []
        K_range = range(1, 11)
        for k_val in K_range:
            kmeans_model_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
            kmeans_model_elbow.fit(kmeans_scaled_data)
            inertia.append(kmeans_model_elbow.inertia_)

        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(K_range, inertia, marker='o', linestyle='--')
        ax_elbow.set_xlabel('Jumlah Cluster (K)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_title('Elbow Method untuk Menentukan K Optimal')
        ax_elbow.set_xticks(K_range)
        ax_elbow.grid(True)
        st.pyplot(fig_elbow)

        k_optimal_input = st.sidebar.number_input("Masukkan Nilai K Optimal (berdasarkan Elbow Plot di atas):", min_value=2, max_value=10, value=3, step=1)

        if st.button("Jalankan K-Means Clustering"):
            final_kmeans_model = KMeans(n_clusters=k_optimal_input, random_state=42, n_init='auto')
            cluster_labels = final_kmeans_model.fit_predict(kmeans_scaled_data)
            
            df_kmeans_interpret = df_kmeans_original_features.copy()
            df_kmeans_interpret['Cluster'] = cluster_labels
            
            st.subheader(f"Karakteristik Rata-Rata per Cluster (K={k_optimal_input})")
            st.markdown("Nilai di bawah adalah rata-rata dari fitur **sebelum di-scaling** (kecuali 'Avg_Time_Social_Media_Encoded' yang sudah di-encode secara ordinal).")
            cluster_characteristics_display = df_kmeans_interpret.groupby('Cluster')[kmeans_feature_names].mean()
            st.dataframe(cluster_characteristics_display)

            if len(kmeans_feature_names) >= 2:
                st.subheader("Visualisasi Cluster Sederhana (2 Fitur Pertama)")
                fig_cluster_scatter, ax_cluster_scatter = plt.subplots(figsize=(10,6))
                scatter = sns.scatterplot(
                    x=kmeans_scaled_data[:, 0], 
                    y=kmeans_scaled_data[:, 1], 
                    hue=cluster_labels, 
                    palette="viridis", 
                    ax=ax_cluster_scatter,
                    s=100, alpha=0.7
                )
                ax_cluster_scatter.set_xlabel(f"Scaled: {kmeans_feature_names[0]}")
                ax_cluster_scatter.set_ylabel(f"Scaled: {kmeans_feature_names[1]}")
                ax_cluster_scatter.set_title(f"Visualisasi Cluster K-Means (K={k_optimal_input}) - 2 Fitur Utama (Scaled)")
                st.pyplot(fig_cluster_scatter)
    else:
        st.warning("Data untuk K-Means tidak dapat diproses.")

    # --- Random Forest Prediction Section ---
    # (Kode untuk Random Forest Prediction tetap sama seperti sebelumnya)
    # Salin bagian Random Forest dari kode Streamlit sebelumnya ke sini
    st.header("🌳 2. Random Forest: Prediksi Status Kesehatan Mental")
    st.markdown("""
    Memprediksi status kesehatan mental (indikasi depresi dan gangguan tidur) dari pola penggunaan media sosial dan data demografis.
    """)
    
    depression_threshold_input = st.sidebar.slider("Threshold untuk 'Merasa Depresi' (skor >= threshold = Ya)", min_value=1, max_value=5, value=4, step=1)
    sleep_threshold_input = st.sidebar.slider("Threshold untuk 'Masalah Tidur' (skor >= threshold = Ya)", min_value=1, max_value=5, value=4, step=1)

    # Di fungsi preprocess_for_rf, pastikan 'preprocessor_rf' yang di-return, bukan 'preprocessor_rf_pipeline'
    # Jika 'preprocessor_rf_pipeline' adalah nama variabel lokal di fungsi itu yang merujuk ke ColumnTransformer, maka OK.
    # Mari kita asumsikan 'preprocessor_rf' adalah nama yang benar untuk ColumnTransformer.
    X, y_depresi, y_tidur, preprocessor_rf = preprocess_for_rf(df_raw.copy(), depression_threshold_input, sleep_threshold_input)


    if X is not None and y_depresi is not None and y_tidur is not None and preprocessor_rf is not None:
        targets_to_predict = {
            "Depresi": y_depresi,
            "Gangguan Tidur": y_tidur
        }

        for target_name, y_target in targets_to_predict.items():
            st.subheader(f"Prediksi untuk: {target_name}")
            
            if y_target.nunique() < 2:
                st.warning(f"Target '{target_name}' hanya memiliki satu kelas setelah binarisasi. Tidak bisa melatih model.")
                continue
            if y_target.isnull().any():
                st.warning(f"Target '{target_name}' memiliki missing values. Tidak bisa melatih model.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.25, random_state=42, stratify=y_target)
            
            if X_train.empty or X_test.empty:
                st.error(f"Dataset training atau testing kosong untuk {target_name} setelah split.")
                continue

            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor_rf),
                                             ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))])
            try:
                model_pipeline.fit(X_train, y_train)
                y_pred = model_pipeline.predict(X_test)

                st.markdown(f"**Akurasi Model {target_name}**: `{accuracy_score(y_test, y_pred):.4f}`")
                st.markdown("**Laporan Klasifikasi:**")
                report_str = classification_report(y_test, y_pred, zero_division=0)
                st.text(report_str)

                st.markdown("**Confusion Matrix:**")
                fig_cm, ax_cm = plt.subplots()
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if target_name == "Depresi" else 'Greens', ax=ax_cm)
                ax_cm.set_title(f'Confusion Matrix - {target_name}')
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)
            
            except ValueError as ve:
                 st.error(f"Terjadi ValueError saat melatih atau mengevaluasi model untuk {target_name}: {ve}")
            except Exception as e_model:
                st.error(f"Error saat melatih model {target_name}: {e_model}")
            st.markdown("---")
    else:
        st.warning("Data untuk Random Forest tidak dapat diproses.")

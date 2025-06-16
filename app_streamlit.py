# app_streamlit_with_auto_persona.py

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
from sklearn.decomposition import PCA # Pastikan ini di-import
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Dashboard Analisis Mental Health & Sosmed", layout="wide")

# --- Fungsi-Fungsi Pembantu ---
@st.cache_data
def load_and_preprocess_data(file_path='smmh.csv'):
    try:
        df = pd.read_csv(file_path)
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
        df['KMeans_Gangguan_Konsentrasi'] = df[['SM_Distraction_Busy', 'Easily_Distracted', 'Difficult_To_Concentrate']].mean(axis=1)
        df['KMeans_Pencarian_Validasi'] = df[['SM_Seek_Validation', 'SM_Compare_Success']].mean(axis=1)
        return df
    except FileNotFoundError:
        st.error(f"🚨 File '{file_path}' tidak ditemukan! Pastikan file tersebut ada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat data dari '{file_path}': {e}")
        return None

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
        st.error(f"Error K-Means preprocessing: {e}")
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
                ('num', StandardScaler(), fitur_rf_numerik)],
            remainder='drop')
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
        return X, y_depresi, y_tidur, preprocessor_rf, kolom_fitur_rf_gabungan
    except Exception as e:
        st.error(f"Error RF preprocessing: {e}")
        return None, None, None, None, None

@st.cache_resource(hash_funcs={ColumnTransformer: lambda x: id(x)})
def get_trained_rf_model(X_full_train, y_full_train, preprocessor, model_name_for_log="Model"):
    if y_full_train.nunique() < 2:
        st.warning(f"Target untuk '{model_name_for_log}' hanya memiliki satu kelas. Model tidak dapat dilatih.")
        return None
    if X_full_train.empty:
        st.warning(f"Data fitur untuk '{model_name_for_log}' kosong. Model tidak dapat dilatih.")
        return None
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))])
    try:
        pipeline.fit(X_full_train, y_full_train)
        # st.success(f"Model untuk '{model_name_for_log}' berhasil dilatih (atau diambil dari cache).") # Aktifkan untuk debug cache
        return pipeline
    except ValueError as ve:
        st.error(f"ValueError saat melatih model '{model_name_for_log}': {ve}")
        return None
    except Exception as e:
        st.error(f"Error umum saat melatih model '{model_name_for_log}': {e}")
        return None

# --- FUNGSI BARU UNTUK PERSONA OTOMATIS ---
def interpretasi_nilai_fitur(nama_fitur, nilai):
    """Helper function untuk mengkategorikan nilai fitur jadi Rendah, Sedang, Tinggi."""
    if nama_fitur == 'Avg_Time_Social_Media_Encoded': 
        if nilai <= 1.5: return "Rendah (Jarang/Singkat)"
        elif nilai <= 3.0: return "Sedang"
        else: return "Tinggi (Sering/Lama)"
    elif nama_fitur in ['SM_No_Purpose', 'KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Bothered_By_Worries']:
        if nilai <= 2.3: return "Rendah"
        elif nilai <= 3.7: return "Sedang"
        else: return "Tinggi"
    return f"{nilai:.2f}" 

def buat_nama_persona_otomatis(df_karakteristik_cluster, k_optimal):
    personas = []
    fitur_kunci = ['Avg_Time_Social_Media_Encoded', 'SM_No_Purpose', 
                   'KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Bothered_By_Worries']

    for i in range(k_optimal):
        if i not in df_karakteristik_cluster.index:
            personas.append({"nama_persona": f"Cluster {i}", "ciri_khas": "Data karakteristik tidak tersedia.", "interpretasi_singkat": ""})
            continue

        cluster_data = df_karakteristik_cluster.loc[i]
        
        # Mengambil nilai dengan aman, antisipasi jika ada fitur yang hilang
        waktu_medsos_val = cluster_data.get('Avg_Time_Social_Media_Encoded', 0) # default 0 jika hilang
        tanpa_tujuan_val = cluster_data.get('SM_No_Purpose', 0)
        gangguan_fokus_val = cluster_data.get('KMeans_Gangguan_Konsentrasi', 0)
        cari_validasi_val = cluster_data.get('KMeans_Pencarian_Validasi', 0)
        cemas_val = cluster_data.get('Bothered_By_Worries', 0)

        waktu_medsos = interpretasi_nilai_fitur('Avg_Time_Social_Media_Encoded', waktu_medsos_val)
        tanpa_tujuan = interpretasi_nilai_fitur('SM_No_Purpose', tanpa_tujuan_val)
        gangguan_fokus = interpretasi_nilai_fitur('KMeans_Gangguan_Konsentrasi', gangguan_fokus_val)
        cari_validasi = interpretasi_nilai_fitur('KMeans_Pencarian_Validasi', cari_validasi_val)
        cemas = interpretasi_nilai_fitur('Bothered_By_Worries', cemas_val)
        
        nama_persona = f"Cluster {i}: Tipe Pengguna Default"
        interpretasi_singkat = "Profil umum, perlu analisis lebih detail dengan melihat ciri khas."

        # Contoh Aturan (Sangat Sederhana, perlu banyak if/elif atau logika lebih canggih)
        if "Tinggi" in waktu_medsos and "Tinggi" in gangguan_fokus and "Tinggi" in cemas:
            nama_persona = f"Cluster {i}: 📱⚡️ Si Aktif Gelisah (Otomatis)"
            interpretasi_singkat = "Cenderung sangat aktif di medsos, namun juga mengalami gangguan fokus dan kecemasan tinggi."
        elif "Rendah" in waktu_medsos and "Rendah" in gangguan_fokus and "Rendah" in cemas and "Rendah" in cari_validasi:
            nama_persona = f"Cluster {i}: 🧘💻 Si Bijak Medsos (Otomatis)"
            interpretasi_singkat = "Penggunaan medsos terkontrol dengan dampak minimal pada fokus, validasi, dan kecemasan."
        elif "Tinggi" in cari_validasi and ("Sedang" in waktu_medsos or "Tinggi" in waktu_medsos):
             nama_persona = f"Cluster {i}: 👍🤳 Si Pencari Pengakuan (Otomatis)"
             interpretasi_singkat = "Fokus utama pada pencarian validasi di media sosial, dengan tingkat aktivitas medsos sedang hingga tinggi."
        elif "Tinggi" in waktu_medsos and "Tinggi" in gangguan_fokus:
            nama_persona = f"Cluster {i}: 🌪️😵 Si Terdistraksi Berat (Otomatis)"
            interpretasi_singkat = "Pengguna berat media sosial yang sangat mudah terganggu konsentrasinya."
        elif "Rendah" in waktu_medsos and "Rendah" in gangguan_fokus and "Tinggi" in cemas :
            nama_persona = f"Cluster {i}: 😟🌿 Si Cemas Meski Jarang Medsos (Otomatis)"
            interpretasi_singkat = "Meskipun jarang menggunakan medsos dan fokusnya baik, tingkat kecemasannya tinggi."


        ciri_khas_list = [
            f"Waktu di Medsos: **{waktu_medsos}** (skor encode: {waktu_medsos_val:.2f})",
            f"Medsos Tanpa Tujuan: **{tanpa_tujuan}** (skor: {tanpa_tujuan_val:.2f})",
            f"Gangguan Konsentrasi: **{gangguan_fokus}** (skor: {gangguan_fokus_val:.2f})",
            f"Pencarian Validasi: **{cari_validasi}** (skor: {cari_validasi_val:.2f})",
            f"Tingkat Kecemasan: **{cemas}** (skor: {cemas_val:.2f})"
        ]
        
        personas.append({
            "nama_persona": nama_persona,
            "ciri_khas": "\n".join([f"- {c}" for c in ciri_khas_list]),
            "interpretasi_singkat": interpretasi_singkat
        })
    return personas


# --- Judul Aplikasi ---
st.title("📊 Dashboard Analisis: Media Sosial & Kesehatan Mental")
st.markdown("Sebuah dashboard interaktif untuk menganalisis data survei terkait penggunaan media sosial dan indikator kesehatan mental.")

# --- Sidebar untuk Input Kontrol ---
st.sidebar.header("⚙️ Pengaturan Analisis")
k_optimal_input = st.sidebar.number_input("Pilih Jumlah Cluster (K) untuk K-Means:", min_value=2, max_value=10, value=3, step=1, help="Tentukan berdasarkan Elbow Plot di tab K-Means.")
depression_threshold_input = st.sidebar.slider("Threshold 'Depresi' (Skor >= Threshold → Ya):", min_value=1, max_value=5, value=4, step=1)
sleep_threshold_input = st.sidebar.slider("Threshold 'Gangguan Tidur' (Skor >= Threshold → Ya):", min_value=1, max_value=5, value=4, step=1)

# --- Memuat Data Langsung ---
DATA_FILE_PATH = 'smmh.csv'
df_raw = load_and_preprocess_data(DATA_FILE_PATH)

if df_raw is None:
    st.error(f"Gagal memuat data dari '{DATA_FILE_PATH}'. Aplikasi tidak dapat melanjutkan.")
    st.stop()
else:
    X_rf_features, y_rf_depresi, y_rf_tidur, preprocessor_rf_obj, rf_feature_names_list = preprocess_for_rf(
        df_raw.copy(), depression_threshold_input, sleep_threshold_input
    )

    tab_list = [
        "🔍 **Ringkasan Data**",
        "🧩 **K-Means Clustering**",
        "🌳 **Random Forest** (Evaluasi Model)",
        "🔮 **Coba Prediksi Sendiri**"
    ]
    tab_overview, tab_kmeans, tab_rf_eval, tab_prediction = st.tabs(tab_list)

    with tab_overview:
        st.header("Ringkasan Data Awal")
        st.markdown(f"Data dimuat dari `{DATA_FILE_PATH}`.")
        col1_ov, col2_ov = st.columns(2)
        with col1_ov: st.metric(label="Jumlah Responden (Baris)", value=f"{df_raw.shape[0]:,}")
        with col2_ov: st.metric(label="Jumlah Variabel Awal (Kolom)", value=f"{df_raw.shape[1]}")
        st.markdown("#### Sampel Data (5 Baris Pertama)")
        st.dataframe(df_raw.head())
        with st.expander("Lihat Statistik Deskriptif Lengkap"): st.dataframe(df_raw.describe(include='all').T)
        with st.expander("Distribusi Beberapa Kolom Penting"):
            st.markdown("##### Distribusi Usia")
            fig_age, ax_age = plt.subplots(); sns.histplot(df_raw['Age'].dropna(), kde=True, ax=ax_age, bins=15); ax_age.set_title("Distribusi Usia Responden"); st.pyplot(fig_age)
            col_gender, col_time = st.columns(2)
            with col_gender:
                st.markdown("##### Distribusi Gender")
                if 'Gender' in df_raw.columns:
                    fig_gender, ax_gender = plt.subplots(); df_raw['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax_gender); ax_gender.set_ylabel(''); ax_gender.set_title("Distribusi Gender"); st.pyplot(fig_gender)
            with col_time:
                st.markdown("##### Waktu Rata-rata di Media Sosial")
                if 'Avg_Time_Social_Media' in df_raw.columns:
                    fig_time, ax_time = plt.subplots(); df_raw['Avg_Time_Social_Media'].value_counts(ascending=True).plot(kind='barh', ax=ax_time); ax_time.set_title("Waktu Rata-rata di Media Sosial"); st.pyplot(fig_time)

    with tab_kmeans:
        st.header("Segmentasi Pengguna dengan K-Means Clustering")
        st.markdown("Mengelompokkan responden berdasarkan pola kebiasaan penggunaan media sosial dan gejala kesehatan mental. Gunakan **input K** di sidebar.")
        kmeans_scaled_data, kmeans_feature_names, df_kmeans_original_features = preprocess_for_kmeans(df_raw.copy())
        
        if kmeans_scaled_data is not None and kmeans_feature_names is not None and df_kmeans_original_features is not None : # Penambahan Cek
            col_elbow, col_k_info = st.columns([2,1])
            with col_elbow:
                st.subheader("Elbow Method untuk Menentukan K Optimal")
                inertia = []; K_range = range(1, 11)
                for k_val_loop in K_range: kmeans_model_elbow = KMeans(n_clusters=k_val_loop, random_state=42, n_init='auto'); kmeans_model_elbow.fit(kmeans_scaled_data); inertia.append(kmeans_model_elbow.inertia_)
                fig_elbow, ax_elbow = plt.subplots(); ax_elbow.plot(K_range, inertia, marker='o', linestyle='--'); ax_elbow.set_xlabel('Jumlah Cluster (K)'); ax_elbow.set_ylabel('Inertia'); ax_elbow.set_xticks(K_range); ax_elbow.grid(True); st.pyplot(fig_elbow)
            with col_k_info: st.info(f"Anda memilih **K = {k_optimal_input}** cluster (lihat sidebar)."); st.markdown("Perhatikan 'siku' pada plot Elbow.")
            
            st.markdown("---"); st.subheader(f"Hasil K-Means Clustering (K={k_optimal_input})")
            final_kmeans_model = KMeans(n_clusters=k_optimal_input, random_state=42, n_init='auto'); cluster_labels = final_kmeans_model.fit_predict(kmeans_scaled_data)
            centroids_scaled = final_kmeans_model.cluster_centers_
            df_kmeans_interpret = df_kmeans_original_features.copy(); df_kmeans_interpret['Cluster'] = cluster_labels
            
            st.markdown("##### Karakteristik Rata-Rata per Cluster"); st.caption("Fitur 'Avg_Time_Social_Media_Encoded' ordinal; fitur lain skor asli/rata-rata.")
            cluster_characteristics_display = df_kmeans_interpret.groupby('Cluster')[kmeans_feature_names].mean()
            st.dataframe(cluster_characteristics_display.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='pink'))

            # --- PENAMBAHAN BAGIAN PERSONA OTOMATIS ---
            if not cluster_characteristics_display.empty:
                st.markdown("---")
                st.subheader("🤖 Saran Persona Otomatis untuk Tiap Cluster")
                st.caption("Nama dan interpretasi di bawah ini dibuat otomatis berdasarkan aturan sederhana. Interpretasi manual Anda tetap yang utama!")
                saran_personas = buat_nama_persona_otomatis(cluster_characteristics_display, k_optimal_input)
                for persona_info in saran_personas:
                    st.markdown(f"#### {persona_info['nama_persona']}")
                    st.markdown("**Ciri Khas Utama (berdasarkan nilai rata-rata fitur K-Means):**")
                    st.markdown(persona_info['ciri_khas'])
                    st.markdown("**Interpretasi Singkat (Saran Otomatis):**")
                    st.markdown(f"> _{persona_info['interpretasi_singkat']}_")
                    st.markdown("---")
            # --- AKHIR BAGIAN PERSONA OTOMATIS ---
            
            st.markdown("---"); st.markdown("#### Visualisasi Cluster")
            if st.checkbox("Tampilkan Visualisasi Cluster dengan PCA (2 Komponen)", value=False):
                st.markdown("##### Plot Scatter: Hasil Reduksi Dimensi dengan PCA (2 Komponen Utama)")
                try:
                    pca = PCA(n_components=2, random_state=42); pca_components = pca.fit_transform(kmeans_scaled_data); pca_centroids = pca.transform(centroids_scaled)
                    fig_pca, ax_pca = plt.subplots(figsize=(10, 6)); sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=cluster_labels, palette=sns.color_palette("viridis", n_colors=k_optimal_input), ax=ax_pca, s=70, alpha=0.7, legend='full'); ax_pca.scatter(pca_centroids[:, 0], pca_centroids[:, 1], marker='X', s=200, color='red', edgecolors='black', label='Centroids'); ax_pca.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)"); ax_pca.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"); ax_pca.set_title(f"Visualisasi Cluster K-Means (K={k_optimal_input}) dengan PCA"); ax_pca.legend(); st.pyplot(fig_pca); st.caption(f"Total varians dijelaskan: {pca.explained_variance_ratio_.sum()*100:.1f}%")
                except Exception as e_pca: st.error(f"Gagal visualisasi PCA: {e_pca}")
        else: st.warning("Data K-Means tidak dapat diproses.")

    with tab_rf_eval:
        st.header("Evaluasi Model Prediksi Random Forest")
        st.markdown("Menilai performa model dalam memprediksi indikasi depresi dan gangguan tidur.")
        if X_rf_features is not None and y_rf_depresi is not None and y_rf_tidur is not None and preprocessor_rf_obj is not None:
            rf_eval_targets = {"Indikasi Depresi": y_rf_depresi, "Indikasi Gangguan Tidur": y_rf_tidur }
            for target_name_eval, y_target_eval in rf_eval_targets.items():
                st.markdown("---"); st.subheader(f"Hasil Evaluasi untuk: {target_name_eval}")
                if y_target_eval.nunique() < 2: st.warning(f"Target '{target_name_eval}' hanya satu kelas."); continue
                if y_target_eval.isnull().any(): st.warning(f"Target '{target_name_eval}' punya missing values."); continue
                X_train, X_test, y_train, y_test = train_test_split(X_rf_features, y_target_eval, test_size=0.25, random_state=42, stratify=y_target_eval)
                if X_train.empty or X_test.empty: st.error(f"Dataset train/test kosong untuk {target_name_eval}."); continue
                model_pipeline_eval = Pipeline(steps=[('preprocessor', preprocessor_rf_obj), ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))])
                try:
                    model_pipeline_eval.fit(X_train, y_train); y_pred_eval = model_pipeline_eval.predict(X_test)
                    col_rf_metric, col_rf_cm = st.columns([1, 1])
                    with col_rf_metric: 
                        st.metric(label=f"Akurasi Model {target_name_eval}", value=f"{accuracy_score(y_test, y_pred_eval):.2%}"); 
                        with st.expander("Laporan Klasifikasi Lengkap"): st.text(classification_report(y_test, y_pred_eval, zero_division=0))
                    with col_rf_cm: 
                        fig_cm_rf, ax_cm_rf = plt.subplots(); cm_rf = confusion_matrix(y_test, y_pred_eval); sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues' if "Depresi" in target_name_eval else 'Greens', ax=ax_cm_rf, xticklabels=model_pipeline_eval.classes_, yticklabels=model_pipeline_eval.classes_); ax_cm_rf.set_title(f'CM - {target_name_eval}'); ax_cm_rf.set_xlabel('Prediksi'); ax_cm_rf.set_ylabel('Aktual'); st.pyplot(fig_cm_rf)
                except ValueError as ve_rf: st.error(f"ValueError model {target_name_eval}: {ve_rf}")
                except Exception as e_model_rf: st.error(f"Error model {target_name_eval}: {e_model_rf}")
        else: st.warning("Data untuk evaluasi Random Forest tidak dapat diproses dengan pengaturan saat ini.")
            
    with tab_prediction:
        st.header("🔮 Coba Prediksi Sendiri Status Kesehatan Mental")
        st.markdown("Masukkan data di bawah ini untuk mendapatkan prediksi berdasarkan model yang telah dilatih.")
        if X_rf_features is not None and y_rf_depresi is not None and y_rf_tidur is not None and preprocessor_rf_obj is not None and rf_feature_names_list is not None:
            gender_options = df_raw['Gender'].dropna().unique().tolist() if 'Gender' in df_raw else ["Male", "Female", "Others", "Prefer not to say"]
            relationship_options = df_raw['Relationship_Status'].dropna().unique().tolist() if 'Relationship_Status' in df_raw else ["Single", "In a relationship", "Married", "Divorced"]
            time_options = ['Less than 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', 'More than 5 hours']
            with st.form("prediction_form"):
                st.markdown("**Masukkan Detail Anda:**")
                col_form1, col_form2 = st.columns(2)
                with col_form1:
                    age_input = st.number_input("Usia Anda:", min_value=10, max_value=100, value=25, step=1)
                    gender_input = st.selectbox("Jenis Kelamin:", options=gender_options, index=0)
                    relationship_input = st.selectbox("Status Hubungan:", options=relationship_options, index=0)
                with col_form2:
                    avg_time_input = st.selectbox("Rata-rata waktu di media sosial per hari:", options=time_options, index=0)
                    sm_no_purpose_input = st.slider("Seberapa sering menggunakan medsos tanpa tujuan spesifik? (1=Sangat Jarang, 5=Sangat Sering)", 1, 5, 3)
                submit_button = st.form_submit_button(label="SUBMIT & PREDIKSI 🚀")

            if submit_button:
                input_data_dict = {
                    'Age': [age_input], 'Gender': [gender_input], 'Relationship_Status': [relationship_input],
                    'Avg_Time_Social_Media': [avg_time_input], 'SM_No_Purpose': [sm_no_purpose_input]
                }
                input_df = pd.DataFrame(input_data_dict, columns=rf_feature_names_list)
                st.markdown("---"); st.subheader("Hasil Prediksi untuk Input Anda:")
                
                model_depresi_live = get_trained_rf_model(X_rf_features, y_rf_depresi, preprocessor_rf_obj, "Depresi (Live Prediction)")
                if model_depresi_live:
                    pred_depresi_live = model_depresi_live.predict(input_df)[0]; proba_depresi_live = model_depresi_live.predict_proba(input_df)[0]
                    hasil_depresi_live = "Berisiko Tinggi 😢" if pred_depresi_live == 1 else "Berisiko Rendah 😊"
                    st.write(f"**Status Indikasi Depresi:** {hasil_depresi_live}")
                    st.progress(float(proba_depresi_live[1]), text=f"Probabilitas Berisiko Tinggi: {proba_depresi_live[1]:.0%}")
                else: st.warning("Model prediksi depresi tidak siap.")

                model_tidur_live = get_trained_rf_model(X_rf_features, y_rf_tidur, preprocessor_rf_obj, "Gangguan Tidur (Live Prediction)")
                if model_tidur_live:
                    pred_tidur_live = model_tidur_live.predict(input_df)[0]; proba_tidur_live = model_tidur_live.predict_proba(input_df)[0]
                    hasil_tidur_live = "Berisiko Tinggi 😴" if pred_tidur_live == 1 else "Berisiko Rendah 🛌"
                    st.write(f"**Status Indikasi Gangguan Tidur:** {hasil_tidur_live}")
                    st.progress(float(proba_tidur_live[1]), text=f"Probabilitas Berisiko Tinggi: {proba_tidur_live[1]:.0%}")
                else: st.warning("Model prediksi gangguan tidur tidak siap.")
        else: st.warning("Data atau komponen untuk prediksi belum siap.")

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Analisis oleh Kelompok 6")
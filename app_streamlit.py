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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.calibration import CalibratedClassifierCV

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
st.title("Dashboard Analisis: Media Sosial & Kesehatan Mental")
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

    st.subheader("Analisis Korelasi dengan Heatmap (Semua Variabel)")
    st.markdown("Heatmap ini mencakup semua variabel yang relevan. Variabel non-numerik (seperti Gender) diubah menjadi angka untuk dianalisis.")

    # Bikin salinan dataframe biar aman
    df_corr_all = df.copy()

    # Looping ke setiap kolom di dataframe
    for col in df_corr_all.columns:
        # Cek apakah tipe data kolom BUKAN angka (int/float)
        if df_corr_all[col].dtype not in ['int64', 'float64']:
            # Jika kolomnya adalah 'Avg_Time_Social_Media', kita pakai OrdinalEncoder
            if col == 'Avg_Time_Social_Media':
                time_categories_corrected = [
                    'Less than an Hour', 'Between 1 and 2 hours', 'Between 2 and 3 hours', 
                    'Between 3 and 4 hours', 'Between 4 and 5 hours', 'More than 5 hours'
                ]
                ordinal_encoder = OrdinalEncoder(categories=[time_categories_corrected])
                df_corr_all[col] = ordinal_encoder.fit_transform(df_corr_all[[col]])
            # Untuk kolom teks lainnya, kita ubah jadi angka dengan pd.factorize
            else:
                df_corr_all[col] = pd.factorize(df_corr_all[col])[0]

    # Setelah semua kolom jadi numerik, hitung korelasinya
    correlation_matrix_all = df_corr_all.corr()

    # Buat heatmap yang lebih besar untuk menampung semua variabel
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(correlation_matrix_all, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, ax=ax, annot_kws={"size": 10})
    plt.title('Heatmap Korelasi Semua Variabel', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)

# ==============================================================================
# TAHAP 2: DATA PREPARATION
# ==============================================================================
st.header("🛠️ Tahap 2: Data Preparation")
st.markdown("Membersihkan, mentransformasi, dan menyiapkan data agar siap untuk diolah oleh model analitik.")

# --- Sidebar untuk Input Kontrol (tetap sama) ---
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
    st.markdown("#### 1. Feature Engineering")
    st.markdown("""
    - **Tujuan**: Membuat fitur baru yang lebih representatif.
    - **Aksi**:
        - `KMeans_Gangguan_Konsentrasi`: Dibuat dengan merata-ratakan skor dari 3 pertanyaan terkait gangguan konsentrasi.
        - `KMeans_Pencarian_Validasi`: Dibuat dengan merata-ratakan skor dari 2 pertanyaan terkait pencarian validasi.
    """)
    st.dataframe(df[['KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi']].head())
    
    st.markdown("---")

    st.markdown("#### 2. Deteksi Missing Value")
    st.markdown("Mengecek apakah ada data yang hilang di setiap kolom.")
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Jumlah Missing': missing_values, 'Persentase (%)': missing_percentage})
    
    # Hanya tampilkan kolom yang punya missing value
    missing_df_filtered = missing_df[missing_df['Jumlah Missing'] > 0]
    
    if missing_df_filtered.empty:
        st.success("✅ Tidak ditemukan missing value pada data.")
    else:
        st.warning("Ditemukan missing value pada kolom berikut:")
        st.dataframe(missing_df_filtered.sort_values(by='Jumlah Missing', ascending=False))
        st.caption("Catatan: Di tahap modeling nanti, missing value pada fitur numerik (jika ada) akan diisi dengan nilai median kolom tersebut.")

    st.markdown("---")

    st.markdown("#### 3. Deteksi Outlier")
    st.markdown("""
    Outlier adalah data yang nilainya jauh berbeda dari sebagian besar data lainnya. Kita bisa mendeteksinya menggunakan **Box Plot**. Titik-titik yang berada di luar "pagar" (garis whisker) pada plot di bawah ini dianggap sebagai outlier.
    """)
    
    # Pilih fitur-fitur kunci yang numerik untuk dideteksi outliernya
    kolom_outlier = ['Age', 'KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Bothered_By_Worries']
    
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(data=df[kolom_outlier], orient='h', palette='Set2', ax=ax)
    ax.set_title('Deteksi Outlier pada Fitur-fitur Kunci', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    st.caption("Catatan: Keberadaan outlier perlu diwaspadai, namun tidak selalu harus dihapus. Terkadang outlier memberikan informasi yang unik. Dalam proyek ini, kita tidak menghapus outlier.")

    st.markdown("---")
    
    st.markdown("#### 4. Transformasi Data Lainnya")
    st.markdown("""
    - **Binarisasi Target**: Mengubah variabel target (`Feel_Depressed`, `Sleep_Issues`) menjadi biner (0 atau 1) berdasarkan _threshold_ yang dipilih di sidebar. Ini diperlukan untuk model klasifikasi.
    - **Encoding & Scaling**: Proses mengubah data kategori menjadi angka (Encoding) dan menyamakan skala fitur (Scaling) adalah langkah krusial. Untuk menjaga integritas evaluasi model, proses ini **sengaja tidak dilakukan di sini**, melainkan akan **dilakukan nanti di dalam _pipeline_ model** pada Tahap Modeling. Ini adalah praktek terbaik untuk mencegah _data leakage_.
    """)


# ==============================================================================
# TAHAP 3: MODELING
# ==============================================================================

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
    for i in range(k_optimal):
        if i not in df_karakteristik_cluster.index:
            continue
        cluster_data = df_karakteristik_cluster.loc[i]
        
        waktu_medsos_val = cluster_data.get('Avg_Time_Social_Media_Encoded', 0)
        gangguan_fokus_val = cluster_data.get('KMeans_Gangguan_Konsentrasi', 0)
        cemas_val = cluster_data.get('Bothered_By_Worries', 0)
        cari_validasi_val = cluster_data.get('KMeans_Pencarian_Validasi', 0)

        waktu_medsos = interpretasi_nilai_fitur('Avg_Time_Social_Media_Encoded', waktu_medsos_val)
        gangguan_fokus = interpretasi_nilai_fitur('KMeans_Gangguan_Konsentrasi', gangguan_fokus_val)
        cemas = interpretasi_nilai_fitur('Bothered_By_Worries', cemas_val)
        cari_validasi = interpretasi_nilai_fitur('KMeans_Pencarian_Validasi', cari_validasi_val)
        
        nama_persona = f"Cluster {i}: Tipe Pengguna Default"
        interpretasi_singkat = "Profil umum, perlu analisis manual lebih detail."

        if "Tinggi" in waktu_medsos and "Tinggi" in gangguan_fokus and "Tinggi" in cemas:
            nama_persona = f"Cluster {i}: 📱⚡️ Si Aktif Gelisah"
            interpretasi_singkat = "Sangat aktif di medsos, namun juga mengalami gangguan fokus dan kecemasan tinggi."
        elif "Rendah" in waktu_medsos and "Rendah" in gangguan_fokus and "Rendah" in cemas:
            nama_persona = f"Cluster {i}: 🧘💻 Si Bijak Medsos"
            interpretasi_singkat = "Penggunaan medsos terkontrol dengan dampak minimal pada fokus dan kecemasan."
        elif "Tinggi" in cari_validasi:
             nama_persona = f"Cluster {i}: 👍🤳 Si Pencari Pengakuan"
             interpretasi_singkat = "Fokus utama pada pencarian validasi di medsos, dengan aktivitas sedang-hingga-tinggi."
        
        ciri_khas_list = [
            f"Waktu di Medsos: **{waktu_medsos}**",
            f"Gangguan Konsentrasi: **{gangguan_fokus}**",
            f"Pencarian Validasi: **{cari_validasi}**",
            f"Tingkat Kecemasan: **{cemas}**"
        ]
        
        personas.append({
            "nama_persona": nama_persona,
            "ciri_khas": "\n".join([f"- {c}" for c in ciri_khas_list]),
            "interpretasi_singkat": interpretasi_singkat
        })
    return personas

st.header("📈 Tahap 3: Modeling")
st.markdown("Membangun model analitik untuk menemukan pola (K-Means) dan membuat prediksi (Logistic Regression).")

with st.expander("Lihat Detail Tahap Modeling & Evaluasi", expanded=False):
    st.subheader("3.1 Modeling Unsupervised: K-Means Clustering")
    st.markdown("Mengelompokkan responden ke dalam beberapa segmen (cluster) berdasarkan kemiripan pola perilaku dan gejala kesehatan mental mereka.")
    
    # Preprocessing untuk K-Means
    try:
        kolom_kmeans_numerik = ['SM_No_Purpose', 'KMeans_Gangguan_Konsentrasi', 'KMeans_Pencarian_Validasi', 'Bothered_By_Worries']
        df_kmeans_selection = df[kolom_kmeans_numerik + ['Avg_Time_Social_Media']].copy()
        
        # Asumsikan 'time_categories_corrected' sudah didefinisikan secara global
        time_categories_corrected = [
            'Less than an Hour', 'Between 1 and 2 hours', 'Between 2 and 3 hours', 
            'Between 3 and 4 hours', 'Between 4 and 5 hours', 'More than 5 hours'
        ]
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
        
        st.markdown("---")
        
        # --- DIAGRAM 1: ELBOW METHOD ---
        st.markdown("##### Diagram 1: Elbow Method untuk Menentukan K Optimal")
        st.caption("Diagram ini membantu kita memilih jumlah cluster (K) yang paling pas. Carilah titik 'siku' (elbow) di mana penurunan 'Inertia' mulai melandai.")
        
        col_elbow1, col_elbow2 = st.columns([2, 1])
        with col_elbow1:
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
            inertia = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans_elbow.fit(kmeans_scaled)
                inertia.append(kmeans_elbow.inertia_)
            ax_elbow.plot(K_range, inertia, marker='o', linestyle='--', color='b')
            ax_elbow.set_xlabel('Jumlah Cluster (K)'); ax_elbow.set_ylabel('Inertia')
            ax_elbow.set_title('Elbow Method'); ax_elbow.set_xticks(K_range); ax_elbow.grid(True)
            st.pyplot(fig_elbow)
        with col_elbow2:
            st.info(f"Anda memilih **K = {k_optimal_input}** cluster (lihat sidebar). Sesuaikan pilihan K di sidebar berdasarkan titik 'siku' pada plot di sebelah.")

        st.markdown("---")
        
        # --- Menjalankan K-Means Final dan Menampilkan Karakteristik ---
        st.markdown(f"##### Hasil Clustering dengan K={k_optimal_input}")
        kmeans_final = KMeans(n_clusters=k_optimal_input, random_state=42, n_init='auto')
        labels = kmeans_final.fit_predict(kmeans_scaled)
        centroids_scaled = kmeans_final.cluster_centers_
        
        centroids_original_scale = scaler_kmeans.inverse_transform(centroids_scaled)
        centroids_df = pd.DataFrame(centroids_original_scale, columns=kolom_kmeans_final)
        
        st.markdown("**Karakteristik Rata-Rata per Cluster (Centroids)**")
        st.dataframe(centroids_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='pink'))
        

        st.markdown("##### Analisis Silhouette K-Means")

        # 1. Prediksi label cluster & hitung silhouette score rata-rata
        labels = kmeans_final.fit_predict(kmeans_scaled)
        sil_score = silhouette_score(kmeans_scaled, labels)

        # Tampilkan score rata-rata dengan st.metric
        st.metric(
            "Silhouette Score (Rata-rata)", 
            f"{sil_score:.4f}", 
            help="Skor rata-rata untuk semua cluster. Semakin mendekati 1, semakin baik."
        )

        # 2. Buat Diagram Silhouette
        st.markdown("Lihat Diagram Silhouette Detail per Cluster")
        # Buat figure untuk plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Hitung silhouette score untuk setiap sample/data
        sample_silhouette_values = silhouette_samples(kmeans_scaled, labels)
        
        y_lower = 10
        n_clusters = kmeans_final.n_clusters
        for i in range(n_clusters):
            # Ambil silhouette score untuk cluster ke-i dan urutkan
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            
            # Tentukan ukuran dan posisi y untuk plot cluster ke-i
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            # Warna untuk cluster
            color = plt.cm.viridis(float(i) / n_clusters)
            
            # Gambar silhouette-nya
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            
            # Beri label nomor cluster di tengah plot-nya
            ax.text(-0.1, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12)
            
            # Update y_lower untuk cluster berikutnya
            y_lower = y_upper + 10  # kasih jarak 10
        # Atur Tampilan Plot
        ax.set_title("Diagram Silhouette untuk Setiap Cluster", fontsize=16)
        ax.set_xlabel("Nilai Koefisien Silhouette", fontsize=12)
        ax.set_ylabel("Label Cluster", fontsize=12)
        
        # Garis vertikal untuk menandai rata-rata silhouette score
        ax.axvline(x=sil_score, color="red", linestyle="--", label='Rata-rata Silhouette Score')
        ax.legend()
        
        ax.set_yticks([])  # Hapus y-axis ticks
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Tampilkan plot di Streamlit
        st.pyplot(fig)

        # --- PENAMBAHAN BAGIAN PERSONA OTOMATIS ---
        st.markdown("---")
        st.subheader("🤖 Saran Persona Otomatis untuk Tiap Cluster")
        st.caption("Nama dan interpretasi di bawah ini dibuat otomatis berdasarkan aturan sederhana. Interpretasi manual Anda tetap yang utama!")

        saran_personas = buat_nama_persona_otomatis(centroids_df, k_optimal_input)

        for persona in saran_personas:
            with st.container(border=True):
                st.markdown(f"#### {persona['nama_persona']}")
                st.markdown("**Ciri Khas Utama (Level Rendah/Sedang/Tinggi):**")
                st.markdown(persona['ciri_khas'])
                st.markdown("**Interpretasi Singkat (Saran Otomatis):**")
                st.markdown(f"> _{persona['interpretasi_singkat']}_")
        
        st.markdown("---")
        
        # --- DIAGRAM 2 & 3: VISUALISASI CLUSTER ---
        st.markdown("##### Diagram 2 & 3: Visualisasi Sebaran Cluster")
        st.caption("Karena data cluster kita memiliki 5 dimensi, kita perlu memvisualisasikannya dalam 2D. Berikut adalah dua cara untuk melihatnya:")

        st.markdown("**Diagram 3: Berdasarkan PCA**")
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(kmeans_scaled)
        pca_centroids = pca.transform(centroids_scaled)

        fig_pca, ax_pca = plt.subplots(figsize=(8, 7))
        sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7, ax=ax_pca)
        sns.scatterplot(x=pca_centroids[:, 0], y=pca_centroids[:, 1], marker='X', s=200, color='red', ax=ax_pca, label='Centroids')
        ax_pca.set_title("Visualisasi Cluster (PCA)")
        ax_pca.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax_pca.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax_pca.legend()
        st.pyplot(fig_pca)

    except Exception as e:
        st.error(f"Terjadi error saat menjalankan K-Means Clustering: {e}")
    
    st.markdown("---")


# ==============================================================================
# TAHAP 4: EVALUATION
# ==============================================================================
st.header("✅ Tahap 4: Evaluation Supervised (Logistic Regression Classifier)")
st.markdown("Mengevaluasi performa dan ketepatan model untuk memastikan model memenuhi tujuan.")

with st.expander("Lihat Detail Tahap Evaluation"):
    # Mendefinisikan fitur dan preprocessor untuk Logistic Regression
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
        # DIUBAH DI SINI
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42))])
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

        # Analisis Variabel Penting (Proxy untuk Logistic Regression)
        st.markdown("**Analisis Variabel Penting**")
        
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
# TAHAP 5: DEPLOYMENT (VERSI TIM SPESIALIS)
# ==============================================================================
st.header("🚀 Tahap 5: Deployment")
st.markdown("Prediksi menggunakan **dua tim fitur spesialis** yang berbeda untuk Depresi dan Gangguan Tidur, berdasarkan pilihan fitur terbaik.")

# --- BAGIAN 1: FORM INPUT SUPER LENGKAP ---
with st.form("prediction_form_spesialis"):
    st.subheader("🔮 Coba Prediksi Sendiri (Mode Spesialis)")
    
    col1, col2 = st.columns(2)
    # Input Demografi Dasar
    with col1:
        age = st.number_input("Usia Anda:", 10, 100, 21, 1)
    with col2:
        time = st.selectbox("Rata-rata waktu di medsos per hari:", time_categories_corrected, index=5)

    st.markdown("---")
    
    # Input Gejala & Perilaku (Skala 1-5)
    st.markdown("**Jawab Pertanyaan Berikut (Skala 1-5):**")
    c1, c2 = st.columns(2)
    with c1:
        bothered = st.slider("Cemas/Khawatir (Bothered by Worries)", 1, 5, 3)
        interest_fluct = st.slider("Minat Naik-Turun (Interest Fluctuation)", 1, 5, 3)
        distracted_busy = st.slider("Terganggu Saat Sibuk (Distracted when Busy)", 1, 5, 3)
        compare_success = st.slider("Membandingkan Diri (Compare to Success)", 1, 5, 3)
    with c2:
        difficult_concentrate = st.slider("Sulit Konsentrasi (Difficult to Concentrate)", 1, 5, 3)
        easily_distracted = st.slider("Mudah Teralihkan (Easily Distracted)", 1, 5, 3)
        seek_validation = st.slider("Mencari Validasi (Seek Validation)", 1, 5, 3)
        no_purpose = st.slider("Medsos Tanpa Tujuan (SM No Purpose)", 1, 5, 3)
    
    submit_spesialis = st.form_submit_button("Dapatkan Prediksi Spesialis")

# --- BAGIAN 2: LOGIKA DUA TIM PREDIKSI ---
if submit_spesialis:

    # --- TIM SPESIALIS 1: DEPRESI ---
    with st.container(border=True):
        st.markdown("#### Hasil Analisis Tim Spesialis Depresi")
        
        # Fitur pilihan lo untuk Depresi + demografi
        fitur_depresi_numerik = [
            'Bothered_By_Worries', 'Difficult_To_Concentrate', 'SM_Seek_Validation',
            'Easily_Distracted', 'Interest_Fluctuation', 'SM_Compare_Success',
            'SM_Distraction_Busy', 'SM_No_Purpose', 'Age'
        ]
        fitur_depresi_ordinal = ['Avg_Time_Social_Media']
        kolom_fitur_depresi = fitur_depresi_numerik + fitur_depresi_ordinal

        # Buat input_df khusus untuk tim ini
        input_df_depresi = pd.DataFrame({
            'Age': [age], 'Avg_Time_Social_Media': [time], 'Bothered_By_Worries': [bothered],
            'Interest_Fluctuation': [interest_fluct], 'Difficult_To_Concentrate': [difficult_concentrate],
            'Easily_Distracted': [easily_distracted], 'SM_Distraction_Busy': [distracted_busy],
            'SM_Compare_Success': [compare_success], 'SM_Seek_Validation': [seek_validation],
            'SM_No_Purpose': [no_purpose]
        })

        preprocessor_depresi = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(categories=[time_categories_corrected]), fitur_depresi_ordinal),
                ('num', StandardScaler(), fitur_depresi_numerik)
            ], remainder='drop'
        )

        
        pipeline_depresi = Pipeline(steps=[('preprocessor', preprocessor_depresi), ('classifier', CalibratedClassifierCV(LogisticRegression(random_state=42), method='isotonic', cv=5))])
        pipeline_depresi.fit(df[kolom_fitur_depresi], df['Target_Depresi'])
        pred_d = pipeline_depresi.predict(input_df_depresi[kolom_fitur_depresi])[0]
        proba_d = pipeline_depresi.predict_proba(input_df_depresi[kolom_fitur_depresi])[0]
        
        hasil_d = "Berisiko Tinggi 😢" if pred_d == 1 else "Berisiko Rendah 😊"
        skor_risiko_d = proba_d[1]
        
        st.metric(label="Keputusan Model", value=hasil_d, delta=f"{skor_risiko_d:.1%} Skor Risiko", delta_color="inverse")
        st.progress(float(skor_risiko_d), text="Risk Meter Depresi")

    # --- TIM SPESIALIS 2: GANGGUAN TIDUR ---
    with st.container(border=True):
        st.markdown("#### Hasil Analisis Tim Spesialis Gangguan Tidur")
        
        # Fitur pilihan lo untuk Gangguan Tidur (tanpa Feel_Depressed) + demografi
        fitur_tidur_numerik = [
            'Interest_Fluctuation', 'Difficult_To_Concentrate', 'Bothered_By_Worries',
            'Easily_Distracted', 'Age'
        ]
        fitur_tidur_ordinal = ['Avg_Time_Social_Media']
        kolom_fitur_tidur = fitur_tidur_numerik + fitur_tidur_ordinal
        
        # Buat input_df khusus untuk tim ini
        input_df_tidur = pd.DataFrame({
            'Age': [age], 'Avg_Time_Social_Media': [time], 'Bothered_By_Worries': [bothered],
            'Interest_Fluctuation': [interest_fluct], 'Difficult_To_Concentrate': [difficult_concentrate],
            'Easily_Distracted': [easily_distracted]
        })
        
        preprocessor_tidur = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(categories=[time_categories_corrected]), fitur_tidur_ordinal),
                ('num', StandardScaler(), fitur_tidur_numerik)
            ], remainder='drop'
        )

        # DIUBAH DI SINI
        pipeline_tidur = Pipeline(steps=[('preprocessor', preprocessor_tidur), ('classifier', CalibratedClassifierCV(LogisticRegression(random_state=42), method='isotonic', cv=5))])
        pipeline_tidur.fit(df[kolom_fitur_tidur], df['Target_Gangguan_Tidur'])
        pred_t = pipeline_tidur.predict(input_df_tidur[kolom_fitur_tidur])[0]
        proba_t = pipeline_tidur.predict_proba(input_df_tidur[kolom_fitur_tidur])[0]
        
        hasil_t = "Berisiko Tinggi 😴" if pred_t == 1 else "Berisiko Rendah 🛌"
        skor_risiko_t = proba_t[1]

        st.metric(label="Keputusan Model", value=hasil_t, delta=f"{skor_risiko_t:.1%} Skor Risiko", delta_color="inverse")
        st.progress(float(skor_risiko_t), text="Risk Meter Gangguan Tidur")
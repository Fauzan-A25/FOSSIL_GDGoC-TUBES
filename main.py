import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load model dan data
model = joblib.load('gb_model.pkl')
data = pd.read_csv('water_potability.csv')

# Splitting features and target
X = data.drop(columns=["Potability"])
y = data["Potability"]

from sklearn.impute import KNNImputer

# Handling missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inisialisasi scaler (optional)
scaler = StandardScaler()
normalizer = MinMaxScaler()

# Setup UI
st.set_page_config(page_title="Fossil Water App", layout="centered", page_icon="💧")

# Navigasi
tabs = st.tabs(["💧 Tentang Aplikasi", "📊 Analisis & Modeling", "🧪 Prediksi Manual"])

# ================================
# TAB 1: Perkenalan Aplikasi
# ================================
with tabs[0]:
    st.title("Fossil - Aplikasi Prediksi Kelayakan Air Minum 💧")
    st.markdown("""
    **Fossil** adalah aplikasi berbasis Machine Learning yang dapat memprediksi apakah air layak minum (potable) atau tidak, 
    berdasarkan parameter kimia dan fisik seperti pH, kadar sulfat, kandungan karbon organik, dan lainnya.

    ### Cara Kerja:
    - Model yang digunakan: **Neural Network (MLPClassifier)**
        - Hidden Layers: (100, 50)
        - Aktivasi: ReLU
        - Optimizer: Adam
        - Iterasi Maksimum: 500
    - Dataset: [Water Potability Dataset](https://www.kaggle.com/adityakadiwal/water-potability)
    - Input: 9 parameter kualitas air
    - Output: Prediksi apakah air tersebut **Potable** atau **Not Potable**

    ---

    ### 👨‍💻 Kelompok Pengembang Aplikasi (Tugas Besar):
    - Fauzan Ahsanudin Alfikri - 103052300003
    - Benedict Joel Purba - 103052300066

    > Aplikasi ini dikembangkan sebagai bagian dari Tugas Besar ***[GDGoC Telkom University]***.
    """)

# ================================
# TAB 2: Visualisasi & Modeling
# ================================
with tabs[1]:
    st.title("📊 Analisis Data Kualitas Air")
    st.write("Berikut ringkasan statistik dan visualisasi dari data:")

    st.subheader("🔍 Cuplikan Data (First 5 Rows)")
    st.dataframe(data.head())

    st.subheader("📈 Statistik Singkat")
    st.dataframe(data.describe())

    st.subheader("📊 Korelasi Antar Fitur")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Distribusi Setiap Fitur")
    selected_feature = st.selectbox("Pilih fitur untuk melihat distribusi:", data.columns[:-1])  # exclude 'Potability'

    fig2, ax2 = plt.subplots()
    sns.histplot(data[selected_feature].dropna(), kde=True, bins=30, color="skyblue", ax=ax2)
    ax2.set_title(f"Distribusi Fitur: {selected_feature}")
    st.pyplot(fig2)

    st.subheader("📦 Boxplot (Deteksi Outlier)")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x=data[selected_feature], color="lightcoral", ax=ax3)
    ax3.set_title(f"Boxplot Fitur: {selected_feature}")
    st.pyplot(fig3)

    st.subheader("🧪 Perbandingan Fitur dengan Potabilitas")
    selected_compare = st.selectbox("Pilih fitur untuk membandingkan dengan 'Potability':", data.columns[:-1])

    fig4, ax4 = plt.subplots()
    sns.violinplot(x="Potability", y=selected_compare, data=data, palette="pastel", ax=ax4)
    ax4.set_xticklabels(["Not Potable", "Potable"])
    ax4.set_title(f"{selected_compare} vs Potability")
    st.pyplot(fig4)



# ================================
# TAB 3: Prediksi Manual
# ================================
with tabs[2]:
    st.title("🧪 Prediksi Kelayakan Air")

    st.markdown("Masukkan nilai untuk masing-masing parameter air:")

    col1, col2, col3 = st.columns(3)

    with col1:
        PH = st.number_input("pH", 0.0, 14.0, step=0.1)
        Hardness = st.number_input("Hardness", 0.0)
        Solids = st.number_input("Solids", 0.0)
    with col2:
        Chloramin = st.number_input("Chloramines", 0.0)
        Sulfate = st.number_input("Sulfate", 0.0)
        Conductivity = st.number_input("Conductivity", 0.0)
    with col3:
        Organic_carbon = st.number_input("Organic Carbon", 0.0)
        Trihalomethanes = st.number_input("Trihalomethanes", 0.0)
        Turbidity = st.number_input("Turbidity", 0.0)

    if st.button("🔍 Prediksi"):
        # Input data
        input_data = pd.DataFrame([[PH, Hardness, Solids, Chloramin, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]],
                                  columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"])
        
        X_train = scaler.fit_transform(X_train)
        X_train = normalizer.fit_transform(X_train)

        # Apply scaling (Standardization + Normalization)
        input_data_scaled = scaler.transform(input_data)  # Apply standardization
        input_data_scaled = normalizer.transform(input_data_scaled)  # Apply normalization

        # Prediction
        prediction = model.predict(input_data_scaled)
        result = "✅ Potable (Layak Minum)" if prediction[0] == 1 else "❌ Not Potable (Tidak Layak Minum)"

        if prediction[0] == 1:
            st.markdown(
                f"""
                <div style="background-color:#d4edda; padding:20px; border-radius:10px; color:#155724;">
                    <h4>Hasil Prediksi: {result}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#f8d7da; padding:20px; border-radius:10px; color:#721c24;">
                    <h4>Hasil Prediksi: {result}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )


import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title='Penguin Classifier', layout='wide')

st.title('🐧 Klasifikasi Penguin Palmer (Naive Bayes)')

st.markdown("""
Aplikasi berbasis web untuk memprediksi jenis penguin menggunakan algoritma **Gaussian Naive Bayes**.
Data yang digunakan diimpor dari `penguins_cleaned.csv`.
""")

@st.cache_data
def load_data(path='penguins_cleaned.csv'):
    df = pd.read_csv(path)
    # Pastikan kolom yang diperlukan ada
    df = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    df = df.reset_index(drop=True)
    return df

@st.cache_data
def train_model(df):
    feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[feature_cols]
    y = df['species']

    # Label encoding species
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0)

    return model, encoder, acc, report, X_train, X_test, y_train, y_test


# load data
try:
    penguins = load_data('penguins_cleaned.csv')
except FileNotFoundError:
    st.error('File penguins_cleaned.csv tidak ditemukan. Pastikan berada di folder yang sama dengan main.py.')
    st.stop()

model, encoder, accuracy, report, X_train, X_test, y_train, y_test = train_model(penguins)

# layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('📊 Preview Data')
    st.dataframe(penguins.head(10))
    st.markdown(f"**Jumlah baris:** {len(penguins)}")

    st.subheader('📈 Statistik Fitur')
    st.dataframe(penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].describe())

with col2:
    st.subheader('📌 Info Model')
    st.metric('Akurasi (test set)', f'{accuracy*100:.2f}%')
    st.text('Classification Report')
    st.code(report)

    st.subheader('🧮 Input untuk Prediksi')
    bill_length = st.number_input('Bill Length (mm)', min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0.0, max_value=30.0, value=14.0, step=0.1)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=300.0, value=200.0, step=1.0)
    body_mass = st.number_input('Body Mass (g)', min_value=0.0, max_value=10000.0, value=4500.0, step=10.0)

    if st.button('🔮 Prediksi Species Penguin'):
        input_data = np.array([[bill_length, bill_depth, flipper_length, body_mass]])
        pred_encoded = model.predict(input_data)[0]
        pred_species = encoder.inverse_transform([pred_encoded])[0]

        st.success(f'Prediksi: **{pred_species}**')
        st.write('Konfigurasi input')
        st.json({
            'bill_length_mm': bill_length,
            'bill_depth_mm': bill_depth,
            'flipper_length_mm': flipper_length,
            'body_mass_g': body_mass
        })


st.markdown('---')

st.subheader('🔍 Analisis distribusi fitur')

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
fig.tight_layout(pad=5.0)

for ax, feature in zip(axs.flatten(), ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']):
    for sp in penguins['species'].unique():
        subset = penguins[penguins['species'] == sp]
        ax.hist(subset[feature], alpha=0.6, bins=15, label=sp)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel('Jumlah')
    ax.legend(fontsize='small')

st.pyplot(fig)

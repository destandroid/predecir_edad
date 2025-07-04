import os
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
from collections import Counter
from datetime import datetime
import plotly.graph_objects as go

# Configuración
st.set_page_config(page_title="Predicción de Edad", layout="wide")

st.title("🔍 Estimador de Edad con CNN")

IMG_SIZE = 100
MODEL_PATH = "copia_utkface_cnn_model.h5"
HISTORIC_FILE = "historico_edades.csv"

# Cargar modelo
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# Etiquetas
age_labels = [f"{i*10}-{i*10+9}" for i in range(9)] + ["90+"]
gender_labels = ["Hombre", "Mujer"]
race_labels = ["Blanco", "Negro", "Asiático", "Indio", "Otros"]

# Inicializar históricos
if os.path.exists(HISTORIC_FILE):
    df_hist = pd.read_csv(HISTORIC_FILE)
    st.session_state["age_history"] = df_hist["edad"].tolist()
    st.session_state["gender_history"] = df_hist["genero"].tolist() if "genero" in df_hist.columns else []
    st.session_state["race_history"] = df_hist["raza"].tolist() if "raza" in df_hist.columns else []
    st.session_state["date_history"] = df_hist["fecha"].tolist() if "fecha" in df_hist.columns else []
else:
    st.session_state["age_history"] = []
    st.session_state["gender_history"] = []
    st.session_state["race_history"] = []
    st.session_state["date_history"] = []

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0
        img_exp = np.expand_dims(img_norm, axis=0)

        age_pred, gender_pred, race_pred = model.predict(img_exp)

        pred_age_index = np.argmax(age_pred)
        pred_age = age_labels[pred_age_index]
        pred_gender = gender_labels[np.argmax(gender_pred)]
        pred_race = race_labels[np.argmax(race_pred)]

        
        col_empty, col_img, col_data = st.columns([0.5, 1, 1])

        with col_img:
            st.image(img_rgb, caption="Imagen subida", width=300)

        with col_data:
            st.markdown(f"### 🧓 Edad estimada: {pred_age}")
            st.markdown(f"### 🚻 Género estimado: {pred_gender}")
            st.markdown(f"### 🌍 Raza estimada: {pred_race}")


        st.session_state["age_history"].append(pred_age)
        st.session_state["gender_history"].append(pred_gender)
        st.session_state["race_history"].append(pred_race)
        st.session_state["date_history"].append(datetime.now().strftime("%Y-%m-%d"))

        # igualar longitudes
        max_len = len(st.session_state["age_history"])
        for key, fill in [
            ("gender_history", "NA"),
            ("race_history", "NA"),
            ("date_history", "NA")
        ]:
            while len(st.session_state[key]) < max_len:
                st.session_state[key].append(fill)

        df_guardar = pd.DataFrame({
            "edad": st.session_state["age_history"],
            "genero": st.session_state["gender_history"],
            "raza": st.session_state["race_history"],
            "fecha": st.session_state["date_history"]
        })
        df_guardar.to_csv(HISTORIC_FILE, index=False)

# Gráficos con Plotly
if st.session_state["age_history"]:
    st.subheader("Histórico de Predicciones")

    # Edad
    age_counts = Counter(st.session_state["age_history"])
    age_fig = go.Figure(data=go.Bar(
        x=age_labels,
        y=[age_counts.get(label, 0) for label in age_labels],
        text=[age_counts.get(label, 0) for label in age_labels],
        textposition="auto"
    ))
    age_fig.update_layout(
        title="Distribución de Edades",
        xaxis_title="Grupo de Edad",
        yaxis_title="Frecuencia",
        yaxis=dict(tickmode='linear', dtick=5),
        height=500
    )

    # Raza
    race_counts = Counter(st.session_state["race_history"])
    race_fig = go.Figure(data=go.Bar(
        x=race_labels,
        y=[race_counts.get(label, 0) for label in race_labels],
        text=[race_counts.get(label, 0) for label in race_labels],
        textposition="auto"
    ))
    race_fig.update_layout(
        title="Distribución por Raza",
        yaxis=dict(tickmode='linear', dtick=10),
        height=500
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(age_fig, use_container_width=True)
    col2.plotly_chart(race_fig, use_container_width=True)

    # Género
    gender_counts = Counter(st.session_state["gender_history"])
    gender_fig = go.Figure(data=go.Pie(
        labels=gender_labels,
        values=[gender_counts.get(label, 0) for label in gender_labels],
        hole=0.3
    ))
    gender_fig.update_layout(
        title="Distribución por Género",
        height=500
    )

    # Peticiones por Día
    date_counts = Counter(st.session_state["date_history"])
    dates = sorted(date_counts.keys())
    y_counts = [date_counts[d] for d in dates]
    formatted_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y") for d in dates]

    peticiones_fig = go.Figure()
    peticiones_fig.add_trace(go.Scatter(
        x=formatted_dates,
        y=y_counts,
        mode='lines+markers',
        line_shape="spline"
    ))
    peticiones_fig.update_layout(
        title="Peticiones por Día",
        xaxis_title="Fecha",
        yaxis_title="Número de Peticiones",
        yaxis=dict(tickmode='linear', dtick=1),
        height=300
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(gender_fig, use_container_width=True)
    col4.plotly_chart(peticiones_fig, use_container_width=True)

# Debe direccionar VS Code a la carpeta con los archivos:
# 1.- Archivo
# 2.- Abrir carpeta. Debe dar click en la carpeta que contiene los archivos de interés
#3.- A la izquierda, en el explorador deberá poder visualizar todos los archivos
#------------------------------------------------------------------------------------------------

# CÓDIGO STREAMLIT
# Ir a:   Ver/Terminal
# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv'): coloca este código
#   python -m venv venv

#---------------------------------------------------------------------------------------
# Luego de crear el ambiente virtual, lo activas
#   .\venv\Scripts\activate   # En Windows
#---------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Cuando vuelva a iniciar sesión, debe volver a activar el ambiente virtual, ya no lo debe crear.
# En este caso debes abrir la carpeta con los archivos del caso.
#---------------------------------------------------------------------------------------------


# Instala la versión específica de scikit-learn
#   pip install scikit-learn==1.2.2
# Instala otras dependencias, incluyendo Streamlit
#  pip install streamlit pandas joblib
#-------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
#      Get-ExecutionPolicy                           Si es Restricted; ejecuta
#      Set-ExecutionPolicy RemoteSigned              Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate


import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# ------------------------- Cargar modelo ------------------------------
regressor = load('Modelopipeline.joblib')

# ------------------------- Opciones ------------------------------
sexo_options = [1, 2]
auto_options = [1, 2, 3, 4, 5]
riqueza_options = [1, 2, 3, 4, 5]
region_options = [1, 2, 3, 4]
binario_options = [0, 1]

# ------------------------- Inicializar ------------------------------
def reset_inputs():
    global edad, riqueza, peso, talla, frutas
    global sexo, autoident, region
    global diabetes, exceso_peso, obesidad

    edad = 0
    riqueza = 1
    peso = 0.0
    talla = 0.0
    frutas = 0
    sexo = 1
    autoident = 1
    region = 1
    diabetes = 0
    exceso_peso = 0
    obesidad = 0

reset_inputs()

# ------------------------- UI ------------------------------
st.title("Modelo de Regresión")
st.markdown("**Predicción de la Presión Arterial Sistólica**")
st.markdown("Basado en variables sociodemográficas y de salud (ENDES 2024. INEI)")
st.markdown("---")
st.markdown("Curso: **Machine Learning en producción - Despliegue web**")
st.markdown("**Autores:**")
st.markdown(".   Julian Apaza")
st.markdown(".   Rodrigo Caballero")
st.markdown(".   Susana Portocarrero")
st.markdown("---")
st.sidebar.header("Ingrese los datos")

# ------------------------- Inputs ------------------------------
edad = st.sidebar.number_input("Edad", min_value=0, value=edad)

riqueza = st.sidebar.selectbox(
    "Nivel de riqueza",
    options=riqueza_options,
    format_func=lambda x: {
        1: "Más pobres",
        2: "Pobre",
        3: "Medio",
        4: "Rico",
        5: "Más rico"
    }[x]
)

peso = st.sidebar.number_input("Peso (kg)", min_value=0.0, value=peso)
talla = st.sidebar.number_input("Talla (m)", min_value=0.0, value=talla)

frutas = st.sidebar.number_input(
    "Porciones de frutas y verduras semanal",
    min_value=0,
    value=frutas
)

sexo = st.sidebar.selectbox(
    "Sexo",
    options=sexo_options,
    format_func=lambda x: "Hombre" if x == 1 else "Mujer"
)

autoident = st.sidebar.selectbox(
    "Autoidentificación",
    options=auto_options,
    format_func=lambda x: {
        1: "Nativo",
        2: "Afro descendiente",
        3: "Blanco",
        4: "Mestizo",
        5: "Otro/No sabe"
    }[x]
)

region = st.sidebar.selectbox(
    "Región",
    options=region_options,
    format_func=lambda x: {
        1: "Lima Metropolitana",
        2: "Resto Costa",
        3: "Sierra",
        4: "Selva"
    }[x]
)

diabetes = st.sidebar.selectbox(
    "¿Tiene diabetes?",
    options=binario_options,
    format_func=lambda x: "Sí" if x == 1 else "No"
)

exceso_peso = st.sidebar.selectbox(
    "¿Tiene exceso de peso?",
    options=binario_options,
    format_func=lambda x: "Sí" if x == 1 else "No"
)

obesidad = st.sidebar.selectbox(
    "¿Presenta obesidad?",
    options=binario_options,
    format_func=lambda x: "Sí" if x == 1 else "No"
)

# ------------------------- Botón Predecir ------------------------------
if st.sidebar.button("Predecir"):

    # Validación básica
    if talla == 0:
        st.error("La talla no puede ser 0")
        st.stop()

    # Crear DataFrame
    obs = pd.DataFrame({
        'EDAD': [edad],
        'RIQUEZA': [riqueza],
        'peso': [peso],
        'talla': [talla],
        'porciones_frutas': [frutas],
        'QSSEXO': [sexo],
        'AUTOIDENTIFICACION': [autoident],
        'SHREGION': [region],
        'diabetes': [diabetes],
        'exceso_peso': [exceso_peso],
        'obesidad': [obesidad]
    })

    # Mostrar inputs (opcional, útil en tesis)
    st.write("Datos ingresados:")
    st.write(obs)

    # Predicción
    pred = regressor.predict(obs)[0]

    # Mostrar resultado
    st.markdown(
        f"<h2 style='color:green;'>Presión Sistólica estimada: {pred:.2f} mmHg</h2>",
        unsafe_allow_html=True
    )

# ------------------------- Reset ------------------------------
if st.sidebar.button("Resetear"):
    reset_inputs()

#   R&D Spend	Administration	Marketing Spend  Ciudad  
#	  142107.34  	91391.77	366168.42         Florida    ---->

# Cambiar los valores.
# Para asignar valores: ver los rangos de las cuantitativas ( MÍNIMO --MÁXIMO)
# eso determinan  cómo predice el modelo. 

#  streamlit run streamlitpipelines.py       en la consola
#  pip freeze > requirements.txt
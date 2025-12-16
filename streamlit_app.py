import streamlit as st
import requests
import json
from datetime import datetime

# Configuración de la API
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="JARVIS RAG System", layout="wide")

st.title("🔍 JARVIS RAG System - Asistente Personal Inteligente")

# Sidebar para navegación
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox("Selecciona el modo", 
                               ["Asistente", "Búsqueda Rápida", "Análisis Profundo", "Gestión de Documentos"])

# Sección de gestión de documentos
if app_mode == "Gestión de Documentos":
    st.header("📁 Gestión de Documentos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Añadir Documento")
        doc_path = st.text_input("Ruta del documento")
        if st.button("Añadir a la cola de procesamiento"):
            if doc_path:
                response = requests.post(f"{API_BASE_URL}/add-document", 
                                       json={"path": doc_path})
                if response.status_code == 200:
                    st.success("Documento añadido correctamente")
                else:
                    st.error("Error al añadir el documento")
            else:
                st.warning("Por favor, introduce una ruta válida")
    
    with col2:
        st.subheader("Documentos Procesados")
        # Aquí iría la lista de documentos procesados

# Sección de búsqueda
elif app_mode in ["Búsqueda Rápida", "Análisis Profundo"]:
    st.header("🔍 Búsqueda Semántica" if app_mode == "Búsqueda Rápida" else "📊 Análisis Profundo")
    
    query = st.text_input("Escribe tu consulta:", 
                         placeholder="Busca en todos tus documentos...")
    
    top_k = st.slider("Número de resultados:", 1, 50, 10)
    
    if st.button("Buscar"):
        if query:
            with st.spinner("Buscando en tus documentos..."):
                response = requests.post(f"{API_BASE_URL}/query", 
                                      json={"query": query, "top_k": top_k})
                
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Búsqueda completada en {results['processing_time']:.2f}s")
                    
                    for i, result in enumerate(results['results']):
                        with st.expander(f"Resultado #{i+1} - Score: {result.get('score', 0):.3f}"):
                            st.write(result.get('text', ''))
                            st.caption(f"Fuente: {result.get('metadata', {}).get('source_document', 'Desconocido')}")
                else:
                    st.error("Error en la búsqueda")
        else:
            st.warning("Por favor, escribe una consulta")

elif app_mode == "Asistente":
    st.header("🧠 Asistente Conversacional")
    query = st.text_area("Escribe tu pregunta:", height=120)
    top_k = st.slider("Número de citas:", 1, 10, 5)
    if st.button("Enviar"):
        if query:
            with st.spinner("Consultando base de conocimiento..."):
                response = requests.post(f"{API_BASE_URL}/chat", 
                                      json={"query": query, "top_k": top_k})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Respuesta")
                    st.write(data.get("answer", ""))
                    st.subheader("Citas")
                    for i, c in enumerate(data.get("citations", [])):
                        with st.expander(f"Cita #{i+1}"):
                            st.write(c.get("text", ""))
                else:
                    st.error("Error en la conversación")
        else:
            st.warning("Por favor, escribe una pregunta")

# Pie de página con información del sistema
st.sidebar.markdown("---")
st.sidebar.subheader("Estado del Sistema")
try:
    health_response = requests.get(f"{API_BASE_URL}/health")
    if health_response.status_code == 200:
        st.sidebar.success("✅ Sistema conectado")
    else:
        st.sidebar.error("❌ Sistema no disponible")
except:
    st.sidebar.error("❌ No se pudo conectar al sistema")

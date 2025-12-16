# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from api.endpoints import router as api_router, orchestrator
from ingestion.watchdog import DocumentWatcher
from database.chroma_db import ChromaDBManager
from ingestion.extractors import DocumentExtractor
from config import DATA_DIR, CHROMA_DB_DIR
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JARVIS-RAG")

app = FastAPI(title="JARVIS RAG System", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(api_router, prefix="/api/v1")

watcher = None

@app.on_event("startup")
async def startup_event():
    """Inicializar el sistema al arrancar la aplicación"""
    logger.info("Iniciando JARVIS RAG System...")
    
    # El monitor y los workers ya se inician en api/endpoints.py
    
    # Iniciar el watcher de documentos en un hilo separado
    global watcher
    watcher = DocumentWatcher(str(DATA_DIR), orchestrator)
    watcher_thread = threading.Thread(target=watcher.start, daemon=True)
    watcher_thread.start()
    
    logger.info("Sistema inicializado correctamente")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al apagar la aplicación"""
    logger.info("Apagando JARVIS RAG System...")
    orchestrator.stop()
    if watcher:
        watcher.stop()
    logger.info("Sistema apagado correctamente")

@app.get("/")
async def root():
    return {"message": "JARVIS RAG System", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "components": {
        "orchestrator": "running",
        "chroma_db": "connected"
    }}

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>JARVIS Assistant</title>
  <style>
    body{font-family:system-ui,Segoe UI,Arial;padding:24px;max-width:900px;margin:0 auto;background:#0f172a;color:#e5e7eb}
    h1{margin:0 0 16px;font-size:24px}
    .card{background:#111827;border:1px solid #1f2937;border-radius:8px;padding:16px;margin-top:12px}
    input,textarea,select,button{width:100%;padding:10px;border-radius:8px;border:1px solid #374151;background:#0b1220;color:#e5e7eb}
    button{background:#2563eb;border:none;cursor:pointer}
    button:hover{background:#1d4ed8}
    .grid{display:grid;gap:12px}
  </style>
  <script>
    async function sendChat(){
      const q=document.getElementById('q').value
      const k=parseInt(document.getElementById('k').value)
      const res=await fetch('/api/v1/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,top_k:k})})
      const data=await res.json()
      document.getElementById('answer').textContent=data.answer||''
      const ctn=document.getElementById('citations'); ctn.innerHTML=''
      (data.citations||[]).forEach((c,i)=>{
        const el=document.createElement('div'); el.className='card'
        el.innerHTML=`<strong>Cita #${i+1}</strong><div>${(c.text||'').replace(/</g,'&lt;')}</div>`
        ctn.appendChild(el)
      })
    }
  </script>
</head>
<body>
  <h1>JARVIS Assistant</h1>
  <div class='grid'>
    <textarea id='q' rows='6' placeholder='Escribe tu pregunta...'></textarea>
    <select id='k'>
      <option value='3'>3</option>
      <option value='5' selected>5</option>
      <option value='8'>8</option>
      <option value='10'>10</option>
    </select>
    <button onclick='sendChat()'>Enviar</button>
  </div>
  <div class='card' style='margin-top:16px'>
    <strong>Respuesta</strong>
    <div id='answer'></div>
  </div>
  <div id='citations'></div>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

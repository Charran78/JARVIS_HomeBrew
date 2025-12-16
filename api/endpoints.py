# api/endpoints.pytos para mas... sientete libre de cambiar la ui y ux a lo que creas que es mas acertado.

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import uuid
import time
import logging
from core.orchestrator import NeuralOrchestrator, TaskType
from models.data_models import Document
from database.chroma_db import ChromaDBManager
from ingestion.extractors import DocumentExtractor
from storage.history_store import HistoryStore
from storage.preferences_store import PreferencesStore
from fastapi.responses import PlainTextResponse
from config import DATA_DIR, CHROMA_DB_DIR
import os

logger = logging.getLogger("API")
router = APIRouter()

# Instancia global del orchestrator
orchestrator = NeuralOrchestrator(ChromaDBManager(str(CHROMA_DB_DIR)), DocumentExtractor())
orchestrator.start_monitor()
orchestrator.start_workers()
history_store = HistoryStore("data/history.db")
preferences_store = PreferencesStore("data/app.db")

class DocumentRequest(BaseModel):
    path: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict] = None

class QueryResponse(BaseModel):
    results: List[Dict]
    processing_time: float

class ChatRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]

@router.post("/add-document")
async def add_document(document: DocumentRequest, background_tasks: BackgroundTasks):
    """Añade un documento para procesar"""
    try:
        background_tasks.add_task(orchestrator.add_document, document.path)
        return {"status": "success", "message": f"Documento {document.path} añadido a la cola de procesamiento"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Realiza una consulta semántica sobre los documentos"""
    try:
        start_time = time.time()
        results = orchestrator.query(request.query, request.top_k, request.filters)
        processing_time = time.time() - start_time
        
        return QueryResponse(results=results, processing_time=processing_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        start_time = time.time()
        result = orchestrator.chat(request.query, request.top_k, request.filters)
        _ = time.time() - start_time
        if request.session_id:
            history_store.append_message(request.session_id, "user", request.query, time.time())
            mid = history_store.append_message_return_id(request.session_id, "assistant", result["answer"], time.time())
            history_store.add_citations(mid, result["citations"])
        return ChatResponse(answer=result["answer"], citations=result["citations"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    try:
        result = orchestrator.chat(request.query, request.top_k, request.filters)
        answer = result["answer"]
        citations = result["citations"]
        async def gen():
            import json
            for i in range(0, len(answer), 64):
                chunk = answer[i:i+64]
                yield (json.dumps({"type": "answer", "text": chunk}) + "\n").encode()
                await asyncio.sleep(0.03)
            yield (json.dumps({"type": "citations", "data": citations}) + "\n").encode()
        from fastapi.responses import StreamingResponse
        if request.session_id:
            history_store.append_message(request.session_id, "user", request.query, time.time())
            mid = history_store.append_message_return_id(request.session_id, "assistant", answer, time.time())
            history_store.add_citations(mid, citations)
        return StreamingResponse(gen(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/start")
async def start_session():
    sid = str(uuid.uuid4())
    history_store.create_session(sid, time.time())
    return {"session_id": sid}

@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    return {"session_id": session_id, "history": history_store.get_history(session_id)}

@router.get("/prefs")
async def get_prefs():
    prefs = preferences_store.get_all()
    return {
        "auto_tts": prefs.get("auto_tts", "false"),
        "tts_rate": prefs.get("tts_rate", "180"),
        "tts_voice_id": prefs.get("tts_voice_id", ""),
        "whisper_model": prefs.get("whisper_model", "base")
    }

class PrefsRequest(BaseModel):
    auto_tts: Optional[bool] = None
    tts_rate: Optional[int] = None
    tts_voice_id: Optional[str] = None
    whisper_model: Optional[str] = None

@router.post("/prefs")
async def set_prefs(req: PrefsRequest):
    prefs = {}
    if req.auto_tts is not None:
        prefs["auto_tts"] = "true" if req.auto_tts else "false"
    if req.tts_rate is not None:
        prefs["tts_rate"] = str(req.tts_rate)
    if req.tts_voice_id is not None:
        prefs["tts_voice_id"] = req.tts_voice_id
    if req.whisper_model is not None:
        prefs["whisper_model"] = req.whisper_model
    if prefs:
        preferences_store.set_many(prefs)
    return {"ok": True}

@router.get("/session/{session_id}/export.md", response_class=PlainTextResponse)
async def export_markdown(session_id: str):
    hist = history_store.get_messages(session_id)
    lines = []
    for m in hist:
        role = m.get("role", "")
        content = m.get("content", "")
        prefix = "👤" if role == "user" else "🤖"
        lines.append(f"{prefix} {content}")
        if role == "assistant":
            cits = history_store.get_citations_for_message(m.get("id"))
            for c in cits:
                lines.append(f"  • {c.get('text','')}")
                meta = c.get('metadata','')
                if meta:
                    lines.append(f"    {meta}")
    return "\n\n".join(lines)

@router.get("/session/{session_id}/export.html", response_class=PlainTextResponse)
async def export_html(session_id: str):
    msgs = history_store.get_messages(session_id)
    parts = ["<html><head><meta charset='utf-8'><title>JARVIS Historial</title></head><body>"]
    parts.append(f"<h1>Historial {session_id}</h1>")
    for m in msgs:
        role = m.get("role","")
        content = m.get("content","")
        parts.append(f"<h3>{'Usuario' if role=='user' else 'Asistente'}</h3>")
        parts.append(f"<div>{content}</div>")
        if role == "assistant":
            cits = history_store.get_citations_for_message(m.get("id"))
            if cits:
                parts.append("<ul>")
                for c in cits:
                    parts.append(f"<li>{c.get('text','')}<br/><small>{c.get('metadata','')}</small></li>")
                parts.append("</ul>")
    parts.append("</body></html>")
    return "".join(parts)

@router.get("/session/{session_id}/export.pdf")
async def export_pdf(session_id: str):
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Times-Roman", 14)
    c.drawString(40, y, f"Historial {session_id}")
    y -= 30
    msgs = history_store.get_messages(session_id)
    c.setFont("Times-Roman", 11)
    for m in msgs:
        role = m.get("role","")
        content = m.get("content","")
        role_title = "Usuario" if role == "user" else "Asistente"
        for line in simpleSplit(f"{role_title}: {content}", "Times-Roman", 11, width - 80):
            if y < 60:
                c.showPage(); y = height - 40; c.setFont("Times-Roman", 11)
            c.drawString(40, y, line); y -= 16
        if role == "assistant":
            cits = history_store.get_citations_for_message(m.get("id"))
            for ct in cits:
                for line in simpleSplit(f"• {ct.get('text','')}", "Times-Roman", 10, width - 100):
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Times-Roman", 10)
                    c.drawString(60, y, line); y -= 14
                meta = ct.get('metadata','')
                if meta:
                    for line in simpleSplit(str(meta), "Times-Roman", 9, width - 110):
                        if y < 60:
                            c.showPage(); y = height - 40; c.setFont("Times-Roman", 9)
                        c.drawString(70, y, line); y -= 12
    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    from fastapi.responses import Response
    return Response(content=pdf, media_type="application/pdf")

class EmailRequest(BaseModel):
    to: str
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True

@router.post("/session/{session_id}/export/email")
async def export_email(session_id: str, req: EmailRequest):
    import smtplib
    from email.mime.text import MIMEText
    md_text = await export_markdown(session_id)
    msg = MIMEText(md_text, "plain", "utf-8")
    msg["Subject"] = f"JARVIS Historial {session_id}"
    msg["From"] = req.username
    msg["To"] = req.to
    server = smtplib.SMTP(req.smtp_server, req.smtp_port)
    try:
        if req.use_tls:
            server.starttls()
        server.login(req.username, req.password)
        server.sendmail(req.username, [req.to], msg.as_string())
    finally:
        server.quit()
    return {"sent": True}

class OAuthEmailRequest(BaseModel):
    provider: str
    access_token: str
    to: str
    subject: Optional[str] = None

@router.post("/session/{session_id}/export/email/oauth")
async def export_email_oauth(session_id: str, req: OAuthEmailRequest):
    import requests as _rq
    md_text = await export_markdown(session_id)
    if req.provider.lower() == "microsoft":
        url = "https://graph.microsoft.com/v1.0/me/sendMail"
        headers = {"Authorization": f"Bearer {req.access_token}", "Content-Type": "application/json"}
        body = {
            "message": {
                "subject": req.subject or f"JARVIS Historial {session_id}",
                "body": {"contentType": "Text", "content": md_text},
                "toRecipients": [{"emailAddress": {"address": req.to}}]
            },
            "saveToSentItems": "true"
        }
        resp = _rq.post(url, headers=headers, json=body)
        return {"status_code": resp.status_code}
    return {"error": "provider_not_supported"}

@router.post("/reindex")
async def reindex():
    base = str(DATA_DIR)
    count = 0
    exts = {".pdf", ".docx", ".txt", ".xlsx"}
    for root, dirs, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            if os.path.splitext(p)[1].lower() in exts:
                orchestrator.add_document(p)
                count += 1
    return {"queued": count}

@router.get("/collections")
async def get_collections():
    """Obtiene información sobre las colecciones disponibles"""
    try:
        stats = orchestrator.chroma_db.get_collection_stats()
        return {"collections": ["documents"], "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def get_files(limit: int = 100, offset: int = 0):
    """Obtiene la lista de documentos procesados"""
    # Implementar lógica para obtener documentos de la base de datos
    return {"files": [], "total": 0, "limit": limit, "offset": offset}

@router.get("/file-details/{document_id}")
async def get_file_details(document_id: str):
    """Obtiene detalles específicos de un documento"""
    # Implementar lógica para obtener detalles del documento
    return {"document_id": document_id, "details": {}}

@router.get("/status")
async def status():
    qs = {
        "EXTRACTION": orchestrator.task_queues[TaskType.EXTRACTION].qsize(),
        "CHUNKING": orchestrator.task_queues[TaskType.CHUNKING].qsize(),
        "EMBEDDING": orchestrator.task_queues[TaskType.EMBEDDING].qsize(),
        "INDEXING": orchestrator.task_queues[TaskType.INDEXING].qsize(),
        "RERANKING": orchestrator.task_queues[TaskType.RERANKING].qsize(),
    }

@router.delete("/delete-document/{document_id}")
async def delete_document(document_id: str):
    try:
        orchestrator.chroma_db.delete_document(document_id)
        return {"deleted": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "active_tasks": {k.name: orchestrator.active_tasks[k] for k in orchestrator.active_tasks},
        "queue_sizes": qs,
        "max_concurrent": {k.name: orchestrator.max_concurrent_tasks[k] for k in orchestrator.max_concurrent_tasks},
        "resources": {
            "cpu": orchestrator.current_cpu,
            "ram": orchestrator.current_ram,
            "gpu": orchestrator.current_gpu,
            "gpu_mem": orchestrator.current_gpu_memory,
        },
        "batch_size": orchestrator.batch_processor.max_batch_size,
        "chroma": orchestrator.chroma_db.get_collection_stats(),
    }

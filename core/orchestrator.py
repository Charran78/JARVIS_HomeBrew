# core/orchestrator.py
import time
import psutil
import queue
from threading import Thread, Event, Lock
import logging
from enum import Enum
import uuid
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import requests
from config import LLM_MODEL
try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NeuralOrchestrator")

class TaskType(Enum):
    EXTRACTION = 1
    CHUNKING = 2
    EMBEDDING = 3
    INDEXING = 4
    RERANKING = 5

@dataclass
class SmartChunk:
    text: str
    chunk_id: str
    source_document: str
    document_id: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None
    batch_id: Optional[str] = None
    processed_time: Optional[float] = None

    def to_dict(self):
        return asdict(self)

class IntelligentBatchProcessor:
    def __init__(self, max_batch_size=1, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.current_batch = []
        self.current_batch_id = str(uuid.uuid4())
        self.last_add_time = time.time()
        self.lock = Lock()
        
    def add_chunk(self, chunk: SmartChunk) -> Optional[List[SmartChunk]]:
        with self.lock:
            self.current_batch.append(chunk)
            self.last_add_time = time.time()
            
            if len(self.current_batch) >= self.max_batch_size:
                return self.process_batch()
            return None
    
    def check_timeout(self) -> Optional[List[SmartChunk]]:
        with self.lock:
            if not self.current_batch:
                return None
                
            if time.time() - self.last_add_time >= self.max_wait_time:
                return self.process_batch()
            return None
    
    def process_batch(self) -> List[SmartChunk]:
        with self.lock:
            batch_to_process = self.current_batch
            self.current_batch = []
            self.current_batch_id = str(uuid.uuid4())
            self.last_add_time = time.time()
            
            # Asignar batch_id a todos los chunks del lote
            for chunk in batch_to_process:
                chunk.batch_id = self.current_batch_id
                
            return batch_to_process

class LoRAManager:
    def __init__(self, base_model=None):
        self.base_model = base_model
        self.lora_adapters = {}
        self.current_adapter = None
        self.lock = Lock()
        
    def apply_lora(self, model, adapter_name="default", r=8, lora_alpha=16, lora_dropout=0.1):
        if not HAS_TORCH:
            logger.error("PyTorch no está disponible para LoRA")
            return model
            
        try:
            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            
            lora_model = get_peft_model(model, lora_config)
            self.lora_adapters[adapter_name] = lora_model
            logger.info(f"LoRA aplicado con éxito: {adapter_name}")
            return lora_model
        except Exception as e:
            logger.error(f"Error aplicando LoRA: {e}")
            return model
    
    def switch_adapter(self, adapter_name):
        with self.lock:
            if adapter_name in self.lora_adapters:
                self.current_adapter = adapter_name
                logger.info(f"Adaptador LoRA cambiado a: {adapter_name}")
            else:
                logger.error(f"Adaptador LoRA no encontrado: {adapter_name}")
    
    def save_adapter(self, adapter_name, path):
        if adapter_name in self.lora_adapters:
            self.lora_adapters[adapter_name].save_pretrained(path)
            logger.info(f"Adaptador {adapter_name} guardado en {path}")
    
    def load_adapter(self, adapter_name, path):
        try:
            # Cargar adaptador LoRA (implementación específica dependiendo del modelo)
            logger.info(f"Adaptador {adapter_name} cargado desde {path}")
        except Exception as e:
            logger.error(f"Error cargando adaptador: {e}")

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
    def initialize(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if HAS_TORCH and torch.cuda.is_available():
                self.model = self.model.cuda()
            self.initialized = True
            logger.info(f"Reranker inicializado: {self.model_name}")
        except Exception as e:
            logger.error(f"Error inicializando reranker: {e}")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[int]:
        if not self.initialized:
            self.initialize()
            if not self.initialized:
                return list(range(min(top_k, len(documents))))
        
        try:
            if not HAS_TORCH:
                return list(range(min(top_k, len(documents))))
            # Preparar inputs para el modelo de reranking
            features = self.tokenizer(
                [query] * len(documents),
                documents,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            if HAS_TORCH and torch.cuda.is_available():
                features = {k: v.cuda() for k, v in features.items()}
            
            # Obtener scores
            with torch.no_grad():
                scores = self.model(**features).logits
            
            # Ordenar documentos por score
            sorted_indices = torch.argsort(scores, descending=True).cpu().numpy().flatten()
            return sorted_indices[:top_k]
            
        except Exception as e:
            logger.error(f"Error en reranking: {e}")
            return list(range(min(top_k, len(documents))))

class NeuralOrchestrator:
    def __init__(self, chroma_db_manager, document_extractor):
        self._stop_event = Event()
        
        # Dependencias
        self.chroma_db = chroma_db_manager
        self.document_extractor = document_extractor
        
        # Umbrales de recursos
        self.cpu_threshold = 80.0
        self.ram_threshold = 80.0
        self.gpu_threshold = 80.0
        self.gpu_memory_threshold = 80.0
        
        # Estado actual de recursos
        self.current_cpu = 0.0
        self.current_ram = 0.0
        self.current_gpu = 0.0
        self.current_gpu_memory = 0.0
        
        # Control de workers y tareas
        self.worker_lock = Lock()
        self.active_tasks = {task_type: 0 for task_type in TaskType}
        self.task_queues = {
            TaskType.EXTRACTION: queue.Queue(),
            TaskType.CHUNKING: queue.Queue(),
            TaskType.EMBEDDING: queue.Queue(),
            TaskType.INDEXING: queue.Queue(),
            TaskType.RERANKING: queue.Queue()
        }
        
        # Configuración dinámica
        self.max_concurrent_tasks = {
            TaskType.EXTRACTION: psutil.cpu_count(logical=False),
            TaskType.CHUNKING: psutil.cpu_count(logical=False),
            TaskType.EMBEDDING: 1,  # Siempre permitir al menos 1 tarea de embedding
            TaskType.INDEXING: 2,
            TaskType.RERANKING: 1
        }
        
        # Procesamiento por lotes
        self.batch_processor = IntelligentBatchProcessor()
        self.batch_lock = Lock()
        
        # Modelos y componentes
        self.lora_manager = LoRAManager()
        self.reranker = Reranker()
        
        # Inicializar NVML para GPU si está disponible
        self.gpu_handles = []
        self.has_gpu = HAS_GPU
        if self.has_gpu:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
                logger.info(f"Detectadas {device_count} GPU(s)")
            except Exception as e:
                logger.error(f"Error inicializando NVML: {e}")
                self.has_gpu = False

    def get_gpu_metrics(self):
        """Obtiene métricas de todas las GPUs disponibles"""
        gpu_metrics = []
        for handle in self.gpu_handles:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_metrics.append({
                    'usage': utilization.gpu,
                    'memory_used': memory_info.used,
                    'memory_total': memory_info.total,
                    'memory_percent': (memory_info.used / memory_info.total) * 100
                })
            except Exception as e:
                logger.error(f"Error obteniendo métricas de GPU: {e}")
                gpu_metrics.append({'usage': 0, 'memory_percent': 0})
        return gpu_metrics

    def monitor_resources(self):
        """Monitorización de recursos en tiempo real"""
        last_update = time.time()
        update_interval = 0.1  # 100ms entre actualizaciones
        
        while not self._stop_event.is_set():
            current_time = time.time()
            if current_time - last_update >= update_interval:
                # Obtener métricas del sistema
                self.current_cpu = psutil.cpu_percent(interval=None)
                self.current_ram = psutil.virtual_memory().percent
                
                # Obtener métricas de GPU
                if self.has_gpu and self.gpu_handles:
                    gpu_metrics = self.get_gpu_metrics()
                    self.current_gpu = max([m['usage'] for m in gpu_metrics])
                    self.current_gpu_memory = max([m['memory_percent'] for m in gpu_metrics])
                else:
                    self.current_gpu = 0.0
                    self.current_gpu_memory = 0.0
                
                # Ajuste dinámico de la capacidad
                self.adjust_capacity()
                
                # Procesar lotes por timeout si es necesario
                batch = self.batch_processor.check_timeout()
                if batch:
                    self.task_queues[TaskType.EMBEDDING].put(batch)
                
                last_update = current_time
            
            time.sleep(0.01)

    def adjust_capacity(self):
        """Ajusta la capacidad dinámica del sistema basado en recursos disponibles"""
        with self.worker_lock:
            # Ajustar capacidad de extracción según CPU disponible
            if self.current_cpu < self.cpu_threshold and self.current_ram < self.ram_threshold:
                self.max_concurrent_tasks[TaskType.EXTRACTION] = min(
                    psutil.cpu_count(logical=False) * 2,
                    self.max_concurrent_tasks[TaskType.EXTRACTION] + 1
                )
            else:
                self.max_concurrent_tasks[TaskType.EXTRACTION] = max(
                    1, 
                    self.max_concurrent_tasks[TaskType.EXTRACTION] - 1
                )
            
            # Ajustar capacidad de embedding según GPU disponible
            if self.has_gpu and self.current_gpu < self.gpu_threshold and self.current_gpu_memory < self.gpu_memory_threshold:
                self.max_concurrent_tasks[TaskType.EMBEDDING] = min(
                    4,
                    self.max_concurrent_tasks[TaskType.EMBEDDING] + 1
                )
            elif self.has_gpu:
                self.max_concurrent_tasks[TaskType.EMBEDDING] = max(
                    1, 
                    self.max_concurrent_tasks[TaskType.EMBEDDING] - 1
                )

    def can_process(self, task_type):
        """Determina si se puede procesar un tipo de tarea específico"""
        with self.worker_lock:
            if self.active_tasks[task_type] >= self.max_concurrent_tasks[task_type]:
                return False
            
            if task_type == TaskType.EXTRACTION:
                return True
            
            elif task_type == TaskType.CHUNKING:
                return True
            
            elif task_type == TaskType.EMBEDDING:
                if not self.has_gpu:
                    return True
                else:
                    return (self.current_gpu < self.gpu_threshold and 
                            self.current_gpu_memory < self.gpu_memory_threshold)
            
            elif task_type == TaskType.INDEXING:
                return True
            
            elif task_type == TaskType.RERANKING:
                return True
        
        return False

    def universal_worker(self, worker_id):
        """Worker polivalente que puede realizar múltiples tipos de tareas"""
        logger.info(f"Iniciando worker universal {worker_id}")
        
        while not self._stop_event.is_set():
            try:
                # Prioridad: embedding > extraction > chunking > indexing > reranking
                for task_type in [TaskType.EMBEDDING, TaskType.EXTRACTION, TaskType.CHUNKING, TaskType.INDEXING, TaskType.RERANKING]:
                    if self.can_process(task_type):
                        try:
                            # Intentar obtener trabajo de la cola correspondiente
                            if task_type == TaskType.EMBEDDING:
                                task = self.task_queues[TaskType.EMBEDDING].get_nowait()
                            elif task_type == TaskType.EXTRACTION:
                                task = self.task_queues[TaskType.EXTRACTION].get_nowait()
                            elif task_type == TaskType.CHUNKING:
                                task = self.task_queues[TaskType.CHUNKING].get_nowait()
                            elif task_type == TaskType.INDEXING:
                                task = self.task_queues[TaskType.INDEXING].get_nowait()
                            else:
                                task = self.task_queues[TaskType.RERANKING].get_nowait()
                            
                            # Registrar que estamos procesando esta tarea
                            with self.worker_lock:
                                self.active_tasks[task_type] += 1
                            
                            # Procesar la tarea
                            if task_type == TaskType.EXTRACTION:
                                result = self.process_extraction(task)
                                # Los chunks van a la cola de embedding
                                for chunk in result:
                                    batch = self.batch_processor.add_chunk(chunk)
                                    if batch:
                                        self.task_queues[TaskType.EMBEDDING].put(batch)
                            elif task_type == TaskType.CHUNKING:
                                result = self.process_chunking(task)
                                # Los chunks van a la cola de embedding
                                for chunk in result:
                                    batch = self.batch_processor.add_chunk(chunk)
                                    if batch:
                                        self.task_queues[TaskType.EMBEDDING].put(batch)
                            elif task_type == TaskType.EMBEDDING:
                                result = self.process_embedding_batch(task)
                                # Los embeddings van a la cola de indexing
                                self.task_queues[TaskType.INDEXING].put(result)
                            elif task_type == TaskType.INDEXING:
                                self.process_indexing(task)
                            else:
                                self.process_reranking(task)
                            
                            # Marcar la tarea como completada
                            self.task_queues[task_type].task_done()
                                
                            # Registrar que hemos terminado esta tarea
                            with self.worker_lock:
                                self.active_tasks[task_type] -= 1
                                
                            break  # Salir del bucle de tipos de tarea
                            
                        except queue.Empty:
                            continue  # No hay trabajo de este tipo
                
                # Si no hay trabajo de ningún tipo, pequeña pausa
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error en worker {worker_id}: {e}")
                with self.worker_lock:
                    # Asegurarse de descontar la tarea activa en caso de error
                    for task_type in TaskType:
                        if self.active_tasks[task_type] > 0:
                            self.active_tasks[task_type] -= 1

    def process_extraction(self, document_info):
        """Procesa la extracción de texto de un documento"""
        logger.info(f"Procesando extracción de: {document_info['path']}")
        
        # Usar el extractor de documentos
        document = self.document_extractor.extract(document_info['path'])
        
        if document and hasattr(document, 'chunks'):
            return document.chunks
        return []

    def process_chunking(self, document):
        """Procesa el troceado de documentos con trazabilidad completa"""
        logger.info(f"Procesando chunking de: {document['path']}")
        
        # Simulación de troceado inteligente con trazabilidad
        chunks = []
        for i in range(3):  # Simular 3 chunks por documento
            chunk = SmartChunk(
                text=f"Contenido del chunk {i} del documento {document['path']}",
                chunk_id=str(uuid.uuid4()),
                source_document=document['path'],
                document_id=document['id'],
                page_number=i // 2,
                paragraph_number=i,
                section_title=f"Sección {i}",
                metadata={
                    "original_path": document['path'],
                    "chunk_index": i,
                    "total_chunks": 3,
                    "processing_timestamp": time.time()
                }
            )
            chunks.append(chunk)
        
        return chunks

    def process_embedding_batch(self, batch):
        """Procesa un lote de chunks para embedding"""
        logger.info(f"Procesando embedding de lote con {len(batch)} chunks")
        
        # Decidir si usar GPU o CPU para embedding
        use_gpu = self.has_gpu and self.current_gpu < self.gpu_threshold and self.current_gpu_memory < self.gpu_memory_threshold
        
        try:
            if use_gpu:
                embeddings = self.gpu_batch_embedding(batch)
            else:
                embeddings = self.cpu_batch_embedding(batch)
                
            # Asignar embeddings a los chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
                chunk.embedding_model = "all-MiniLM-L6-v2"
                chunk.embedding_version = "1.0"
                chunk.processed_time = time.time()
                
            return batch
            
        except Exception as e:
            logger.error(f"Error en embedding de lote: {e}")
            # Fallback a CPU si la GPU falla
            if use_gpu:
                logger.info("Intentando fallback a CPU para embedding de lote")
                embeddings = self.cpu_batch_embedding(batch)
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                    chunk.embedding_model = "all-MiniLM-L6-v2"
                    chunk.embedding_version = "1.0"
                    chunk.processed_time = time.time()
                return batch
            raise

    def gpu_batch_embedding(self, batch):
        """Embedding por lotes usando GPU"""
        # Simulación de embedding con GPU
        time.sleep(0.01 * len(batch))
        return [[0.1 * i for i in range(384)] for _ in range(len(batch))]

    def cpu_batch_embedding(self, batch):
        """Embedding por lotes usando CPU"""
        try:
            import mmh3
        except Exception:
            time.sleep(0.01 * len(batch))
            return [[0.0 for _ in range(384)] for _ in range(len(batch))]
        emb_dim = 384
        out = []
        for chunk in batch:
            text = (getattr(chunk, 'text', '') or '').lower()
            tokens = [t for t in text.split() if t]
            vec = [0.0] * emb_dim
            for idx, tok in enumerate(tokens[:1024]):
                h = mmh3.hash(tok, seed=idx % 17, signed=False)
                pos = h % emb_dim
                vec[pos] += 1.0
            s = sum(vec) or 1.0
            vec = [v / s for v in vec]
            out.append(vec)
        return out

    def process_indexing(self, batch):
        """Procesa indexación de un lote de embeddings"""
        logger.info(f"Indexando lote con {len(batch)} embeddings")
        
        # Guardar en ChromaDB
        self.chroma_db.add_embeddings(batch)
        
        # Guardar trazabilidad completa en base de datos
        for chunk in batch:
            self.save_chunk_metadata(chunk)

    def process_reranking(self, task):
        """Procesa reranking de resultados de búsqueda"""
        query, documents = task
        logger.info(f"Reranking para query: {query} con {len(documents)} documentos")
        
        # Extraer textos para reranking
        texts = [doc['text'] for doc in documents]
        
        # Aplicar reranking
        ranked_indices = self.reranker.rerank(query, texts)
        
        # Reordenar documentos según reranking
        ranked_documents = [documents[i] for i in ranked_indices]
        
        return ranked_documents

    def save_chunk_metadata(self, chunk):
        """Guarda los metadatos del chunk para trazabilidad completa"""
        # En una implementación real, esto guardaría en una base de datos
        metadata = chunk.to_dict()
        logger.debug(f"Guardando metadatos del chunk: {chunk.chunk_id}")

    def add_document(self, document_path):
        """Añade un documento para procesar"""
        document = {"path": document_path, "id": str(uuid.uuid4())}
        self.task_queues[TaskType.EXTRACTION].put(document)
        logger.info(f"Documento añadido a la cola: {document_path}")

    def query(self, query_text, top_k=10, filters=None):
        try:
            qchunk = SmartChunk(text=query_text, chunk_id=str(uuid.uuid4()), source_document="query", document_id="query")
            qemb = self.cpu_batch_embedding([qchunk])[0]
            res = self.chroma_db.query(qemb, n_results=top_k, where=filters)
            docs = []
            if isinstance(res, dict) and "documents" in res:
                doc_lists = res.get("documents", [])
                meta_lists = res.get("metadatas", [])
                id_lists = res.get("ids", [])
                if doc_lists:
                    dl = doc_lists[0]
                    ml = meta_lists[0] if meta_lists else []
                    il = id_lists[0] if id_lists else []
                    for i, d in enumerate(dl):
                        md = ml[i] if i < len(ml) else {}
                        docs.append({"text": d, "score": None, "metadata": md})
            if not docs:
                docs = [{"text": "Sin resultados", "score": None, "metadata": {}}]
            self.task_queues[TaskType.RERANKING].put((query_text, docs))
            return docs[:top_k]
        except Exception:
            retrieved_docs = [
                {"text": f"Documento {i}", "score": 0.9 - (i * 0.1), "metadata": {}}
                for i in range(top_k)
            ]
            return retrieved_docs

    def chat(self, query_text, top_k=10, filters=None):
        docs = self.query(query_text, top_k, filters)
        ctx = "\n\n".join([f"[{i+1}] {d.get('text','')}" for i, d in enumerate(docs)])
        sysmsg = "Responde en español usando exclusivamente el contexto."
        prompt = f"Pregunta: {query_text}\n\nContexto:\n{ctx}\n\nRespuesta:"
        answer = None
        try:
            url = "http://127.0.0.1:11434/api/generate"
            payload = {"model": LLM_MODEL, "prompt": f"{sysmsg}\n\n{prompt}", "stream": False}
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 200:
                j = r.json()
                answer = j.get("response", "").strip()
        except Exception:
            answer = None
        if not answer:
            answer = " ".join([d.get("text", "") for d in docs]) or "Sin resultados"
        citations = [{"text": d.get("text", ""), "score": d.get("score", None), "metadata": d.get("metadata", {})} for d in docs]
        return {"answer": answer, "citations": citations}

    def start_workers(self, num_workers=None):
        """Inicia los workers del sistema"""
        if num_workers is None:
            num_workers = psutil.cpu_count(logical=True)
        
        for i in range(num_workers):
            worker = Thread(target=self.universal_worker, args=(f"worker-{i}",))
            worker.daemon = True
            worker.start()
            logger.info(f"Iniciado worker {i}")

    def start_monitor(self):
        """Inicia el monitor de recursos"""
        monitor_thread = Thread(target=self.monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("Monitor de recursos iniciado")

    def stop(self):
        """Detiene el sistema"""
        self._stop_event.set()
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        logger.info("Sistema detenido")

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancias de dependencias (en una implementación real se inyectarían)
    from database.chroma_db import ChromaDBManager
    from ingestion.extractors import DocumentExtractor
    
    chroma_db = ChromaDBManager()
    extractor = DocumentExtractor()
    
    orchestrator = NeuralOrchestrator(chroma_db, extractor)
    
    # Iniciar el sistema
    orchestrator.start_monitor()
    orchestrator.start_workers()
    
    # Añadir algunos documentos de ejemplo
    for i in range(5):
        orchestrator.add_document(f"/path/to/document_{i}.txt")
    
    # Esperar a que se procesen las tareas
    try:
        while True:
            time.sleep(1)
            # Mostrar estado actual
            with orchestrator.worker_lock:
                status = f"Estado: Extraction={orchestrator.active_tasks[TaskType.EXTRACTION]}, "
                status += f"Chunking={orchestrator.active_tasks[TaskType.CHUNKING]}, "
                status += f"Embedding={orchestrator.active_tasks[TaskType.EMBEDDING]}, "
                status += f"Indexing={orchestrator.active_tasks[TaskType.INDEXING]}, "
                status += f"Reranking={orchestrator.active_tasks[TaskType.RERANKING]}, "
                status += f"CPU={orchestrator.current_cpu:.1f}%, "
                status += f"RAM={orchestrator.current_ram:.1f}%"
                if HAS_GPU:
                    status += f", GPU={orchestrator.current_gpu:.1f}%"
                logger.info(status)
    except KeyboardInterrupt:
        orchestrator.stop()

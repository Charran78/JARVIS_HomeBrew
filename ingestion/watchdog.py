# ingestion/watchdog.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from core.orchestrator import NeuralOrchestrator

logger = logging.getLogger("DocumentWatcher")

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, orchestrator: NeuralOrchestrator):
        self.orchestrator = orchestrator
        
    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"New file detected: {event.src_path}")
            self.orchestrator.add_document(event.src_path)
            
    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            self.orchestrator.add_document(event.src_path)
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            # Implementar lógica para eliminar documento de la base de datos

class DocumentWatcher:
    def __init__(self, watch_path: str, orchestrator: NeuralOrchestrator):
        self.watch_path = watch_path
        self.orchestrator = orchestrator
        self.observer = Observer()
        self.event_handler = DocumentHandler(orchestrator)
        
    def start(self):
        self.observer.schedule(self.event_handler, self.watch_path, recursive=True)
        self.observer.start()
        logger.info(f"Started watching directory: {self.watch_path}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
        
        self.observer.join()
    
    def stop(self):
        self.observer.stop()
        self.observer.join()
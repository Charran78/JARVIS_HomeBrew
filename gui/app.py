import sys, json
import requests
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, QSpinBox, QListWidget, QListWidgetItem, QCheckBox, QComboBox, QSlider, QLineEdit
from PySide6.QtCore import Qt, Signal, QThread, QBuffer, QEventLoop, QTimer, QIODevice
from PySide6.QtGui import QShortcut, QKeySequence, QTextCursor
from PySide6.QtMultimedia import QMediaDevices, QAudioFormat, QAudioSource

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

class ChatWorker(QThread):
    finished = Signal(dict)
    def __init__(self, query, top_k, session_id, parent=None):
        super().__init__(parent)
        self.query = query
        self.top_k = top_k
        self.session_id = session_id
    def run(self):
        try:
            r = requests.post(f"{API_BASE_URL}/chat", json={"query": self.query, "top_k": self.top_k, "session_id": self.session_id})
            data = r.json() if r.status_code == 200 else {"answer": "Error", "citations": []}
            self.finished.emit(data)
        except Exception:
            self.finished.emit({"answer": "Error", "citations": []})

class StreamWorker(QThread):
    answerChunk = Signal(str)
    citationsDone = Signal(list)
    def __init__(self, query, top_k, session_id, parent=None):
        super().__init__(parent)
        self.query = query
        self.top_k = top_k
        self.session_id = session_id
    def run(self):
        try:
            with requests.post(f"{API_BASE_URL}/chat-stream", json={"query": self.query, "top_k": self.top_k, "session_id": self.session_id}, stream=True) as r:
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") == "answer":
                        self.answerChunk.emit(obj.get("text", ""))
                    elif obj.get("type") == "citations":
                        self.citationsDone.emit(obj.get("data", []))
        except Exception:
            pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS Assistant")
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QVBoxLayout(cw)
        layout.addWidget(QLabel("Pregunta"))
        self.queryEdit = QTextEdit()
        self.queryEdit.setPlaceholderText("Escribe tu pregunta...")
        layout.addWidget(self.queryEdit)
        topRow = QHBoxLayout()
        topRow.addWidget(QLabel("Citas"))
        self.kSpin = QSpinBox()
        self.kSpin.setRange(1, 10)
        self.kSpin.setValue(5)
        topRow.addWidget(self.kSpin)
        layout.addLayout(topRow)
        self.sendBtn = QPushButton("Enviar (Ctrl+Enter)")
        layout.addWidget(self.sendBtn)
        layout.addWidget(QLabel("Respuesta"))
        self.answerEdit = QTextEdit()
        self.answerEdit.setReadOnly(True)
        layout.addWidget(self.answerEdit)
        speakRow = QHBoxLayout()
        self.speakAuto = QCheckBox("Hablar automáticamente")
        self.speakBtn = QPushButton("🔈 Hablar (Alt+S)")
        speakRow.addWidget(self.speakAuto)
        speakRow.addWidget(self.speakBtn)
        layout.addLayout(speakRow)
        voiceRow = QHBoxLayout()
        voiceRow.addWidget(QLabel("Voz"))
        self.voiceCombo = QComboBox()
        voiceRow.addWidget(self.voiceCombo)
        rateRow = QHBoxLayout()
        rateRow.addWidget(QLabel("Velocidad"))
        self.rateSlider = QSlider(Qt.Horizontal)
        self.rateSlider.setMinimum(120)
        self.rateSlider.setMaximum(240)
        self.rateSlider.setValue(180)
        rateRow.addWidget(self.rateSlider)
        layout.addLayout(voiceRow)
        layout.addLayout(rateRow)
        layout.addWidget(QLabel("Citas"))
        self.citationsList = QListWidget()
        layout.addWidget(self.citationsList)
        layout.addWidget(QLabel("Historial"))
        self.historyList = QListWidget()
        layout.addWidget(self.historyList)
        self.micBtn = QPushButton("🎙️ Mic (Alt+M)")
        self.micWhisperBtn = QPushButton("🌀 Whisper (Alt+W)")
        layout.addWidget(self.micBtn)
        layout.addWidget(self.micWhisperBtn)
        exportRow = QHBoxLayout()
        self.copyMdBtn = QPushButton("📋 Copiar Markdown")
        exportRow.addWidget(self.copyMdBtn)
        self.emailTo = QLineEdit()
        self.emailTo.setPlaceholderText("destinatario@example.com")
        exportRow.addWidget(self.emailTo)
        self.emailServer = QLineEdit()
        self.emailServer.setPlaceholderText("smtp.example.com")
        exportRow.addWidget(self.emailServer)
        self.emailPort = QLineEdit()
        self.emailPort.setPlaceholderText("587")
        exportRow.addWidget(self.emailPort)
        self.emailUser = QLineEdit()
        self.emailUser.setPlaceholderText("usuario")
        exportRow.addWidget(self.emailUser)
        self.emailPass = QLineEdit()
        self.emailPass.setEchoMode(QLineEdit.Password)
        self.emailPass.setPlaceholderText("clave")
        exportRow.addWidget(self.emailPass)
        self.sendEmailBtn = QPushButton("✉️ Enviar")
        exportRow.addWidget(self.sendEmailBtn)
        layout.addLayout(exportRow)
        self.sendBtn.clicked.connect(self.onSend)
        self.micBtn.clicked.connect(self.onMic)
        self.micWhisperBtn.clicked.connect(self.onMicWhisper)
        self.speakBtn.clicked.connect(self.onSpeak)
        self.voiceCombo.currentIndexChanged.connect(self.onPrefsChange)
        self.rateSlider.valueChanged.connect(self.onPrefsChange)
        self.speakAuto.stateChanged.connect(self.onPrefsChange)
        self.copyMdBtn.clicked.connect(self.onCopyMd)
        self.sendEmailBtn.clicked.connect(self.onSendEmail)
        self.setStyleSheet("QWidget{background:#0f172a;color:#e5e7eb;} QTextEdit,QSpinBox{background:#0b1220;border:1px solid #374151;border-radius:8px;padding:8px;} QPushButton{background:#2563eb;color:#fff;border:none;padding:10px;border-radius:8px;} QPushButton:hover{background:#1d4ed8;} QLabel{font-weight:bold;margin-top:10px;} QListWidget{background:#0b1220;border:1px solid #374151;border-radius:8px;}")
        # Sesión
        self.session_id = None
        self.whisper_model_name = None
        self.whisper_model = None
        try:
            res = requests.post(f"{API_BASE_URL}/session/start")
            if res.status_code == 200:
                self.session_id = res.json().get("session_id")
        except Exception:
            pass
        try:
            r = requests.get(f"{API_BASE_URL}/prefs")
            if r.status_code == 200:
                d = r.json()
                self.speakAuto.setChecked(d.get("auto_tts", "false") == "true")
                self.rateSlider.setValue(int(d.get("tts_rate", "180")))
        except Exception:
            pass
        try:
            import pyttsx3
            e = pyttsx3.init()
            voices = e.getProperty('voices')
            self.voiceCombo.clear()
            for v in voices:
                self.voiceCombo.addItem(v.name, v.id)
            r = requests.get(f"{API_BASE_URL}/prefs")
            if r.status_code == 200:
                vid = r.json().get("tts_voice_id", "")
                if vid:
                    for i in range(self.voiceCombo.count()):
                        if self.voiceCombo.itemData(i) == vid:
                            self.voiceCombo.setCurrentIndex(i)
                            break
        except Exception:
            pass
        # Atajos
        QShortcut(QKeySequence("Ctrl+Return"), self, self.onSend)
        QShortcut(QKeySequence("Alt+M"), self, self.onMic)
        QShortcut(QKeySequence("Alt+W"), self, self.onMicWhisper)
        QShortcut(QKeySequence("Alt+S"), self, self.onSpeak)
        QShortcut(QKeySequence("Ctrl+Up"), self, lambda: self.kSpin.setValue(min(self.kSpin.value()+1, self.kSpin.maximum())))
        QShortcut(QKeySequence("Ctrl+Down"), self, lambda: self.kSpin.setValue(max(self.kSpin.value()-1, self.kSpin.minimum())))
    def record_wav(self, path: str, duration_ms: int = 5000, sample_rate: int = 16000):
        devices = QMediaDevices.audioInputs()
        if not devices:
            raise RuntimeError("no_audio_device")
        dev = devices[0]
        fmt = QAudioFormat()
        fmt.setSampleRate(sample_rate)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        src = QAudioSource(dev, fmt)
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        src.start(buf)
        loop = QEventLoop()
        QTimer.singleShot(duration_ms, loop.quit)
        loop.exec()
        src.stop()
        buf.close()
        import wave
        raw = bytes(buf.data())
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
        wf.close()
    def onSend(self):
        q = self.queryEdit.toPlainText().strip()
        if not q:
            return
        self.sendBtn.setEnabled(False)
        self.answerEdit.clear()
        self.citationsList.clear()
        # Añadir al historial (user)
        self.historyList.addItem(QListWidgetItem("👤 " + q))
        # Streaming
        self.worker = StreamWorker(q, self.kSpin.value(), self.session_id, self)
        self.worker.answerChunk.connect(self.onChunk)
        self.worker.citationsDone.connect(self.onCitations)
        self.worker.start()
    def onResult(self, data):
        self.answerEdit.setPlainText(data.get("answer", ""))
        self.citationsList.clear()
        for c in data.get("citations", []):
            item = QListWidgetItem(c.get("text", ""))
            self.citationsList.addItem(item)
        self.sendBtn.setEnabled(True)
        # Añadir al historial (assistant)
        self.historyList.addItem(QListWidgetItem("🤖 " + data.get("answer", "")))
    def onChunk(self, text):
        self.answerEdit.moveCursor(QTextCursor.End)
        self.answerEdit.insertPlainText(text)
    def onCitations(self, citations):
        self.citationsList.clear()
        for c in citations:
            item = QListWidgetItem(c.get("text", ""))
            self.citationsList.addItem(item)
        self.sendBtn.setEnabled(True)
        # Añadir al historial (assistant)
        self.historyList.addItem(QListWidgetItem("🤖 " + self.answerEdit.toPlainText()))
        if self.speakAuto.isChecked():
            self.onSpeak()
    def onMic(self):
        try:
            try:
                self.record_wav('tmp_google.wav', duration_ms=5000, sample_rate=16000)
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile('tmp_google.wav') as source:
                    audio = r.record(source)
                try:
                    text = r.recognize_google(audio, language="es-ES")
                    self.queryEdit.setPlainText(text)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            self.micBtn.setText("🎙️ Mic no disponible")
    def onMicWhisper(self):
        try:
            try:
                self.record_wav('tmp_whisper.wav', duration_ms=5000, sample_rate=16000)
            except Exception:
                raise
            model_name = "base"
            try:
                r = requests.get(f"{API_BASE_URL}/prefs")
                if r.status_code == 200:
                    d = r.json()
                    mn = d.get("whisper_model", "base")
                    if isinstance(mn, str) and mn:
                        model_name = mn
            except Exception:
                pass
            try:
                try:
                    from faster_whisper import WhisperModel
                    if self.whisper_model_name != model_name or self.whisper_model is None:
                        self.whisper_model = WhisperModel(model_name, device="cpu")
                        self.whisper_model_name = model_name
                    segments, info = self.whisper_model.transcribe('tmp_whisper.wav', language='es')
                    text = " ".join([s.text for s in segments])
                    result = {"text": text}
                except Exception:
                    import whisper
                    if self.whisper_model_name != model_name or self.whisper_model is None:
                        self.whisper_model = whisper.load_model(model_name)
                        self.whisper_model_name = model_name
                    result = self.whisper_model.transcribe('tmp_whisper.wav', language='es')
            except Exception:
                result = {"text": ""}
            txt = result.get('text', '').strip()
            if txt:
                self.queryEdit.setPlainText(txt)
        except Exception:
            self.micWhisperBtn.setText("🌀 Whisper no disponible")
    def onSpeak(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', int(self.rateSlider.value()))
            vid = self.voiceCombo.currentData()
            if vid:
                engine.setProperty('voice', vid)
            engine.say(self.answerEdit.toPlainText())
            engine.runAndWait()
        except Exception:
            self.speakBtn.setText("🔈 TTS no disponible")
    def onPrefsChange(self):
        try:
            data = {
                "auto_tts": self.speakAuto.isChecked(),
                "tts_rate": int(self.rateSlider.value()),
                "tts_voice_id": self.voiceCombo.currentData() or ""
            }
            requests.post(f"{API_BASE_URL}/prefs", json=data)
        except Exception:
            pass
    def onCopyMd(self):
        try:
            r = requests.get(f"{API_BASE_URL}/session/{self.session_id}/export.md")
            if r.status_code == 200:
                QApplication.clipboard().setText(r.text)
        except Exception:
            pass
    def onSendEmail(self):
        try:
            server = self.emailServer.text().strip()
            port = int(self.emailPort.text().strip() or "587")
            to = self.emailTo.text().strip()
            user = self.emailUser.text().strip()
            pw = self.emailPass.text()
            data = {
                "to": to,
                "smtp_server": server,
                "smtp_port": port,
                "username": user,
                "password": pw,
                "use_tls": True
            }
            requests.post(f"{API_BASE_URL}/session/{self.session_id}/export/email", json=data)
        except Exception:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 700)
    w.show()
    sys.exit(app.exec())

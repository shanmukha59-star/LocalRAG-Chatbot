import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PySide6.QtCore import QThread, Signal
from rag_engine import ask_rag, ensure_models_downloaded


class ModelLoaderThread(QThread):
    status_update = Signal(str)
    finished_loading = Signal()
    failed_loading = Signal(str)

    def run(self):
        try:
            ensure_models_downloaded(progress_callback=self.status_update.emit)
            self.finished_loading.emit()
        except Exception as error:
            self.failed_loading.emit(str(error))


class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Chatbot")
        self.resize(600, 500)

        layout = QVBoxLayout()
        self.status_label = QLabel("Preparing app, please wait...")
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_box = QLineEdit()
        self.send_button = QPushButton("Send")

        layout.addWidget(self.status_label)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.input_box)
        layout.addWidget(self.send_button)
        self.setLayout(layout)

        self.send_button.clicked.connect(self.handle_send)
        self.history = []

        # Disable input until models are ready
        self.input_box.setEnabled(False)
        self.send_button.setEnabled(False)

        self.loader_thread = ModelLoaderThread()
        self.loader_thread.status_update.connect(self.status_label.setText)
        self.loader_thread.finished_loading.connect(self.on_models_ready)
        self.loader_thread.failed_loading.connect(self.on_models_failed)
        self.loader_thread.start()

    def on_models_ready(self):
        self.status_label.setText("Ready! Ask a question below.")
        self.input_box.setEnabled(True)
        self.send_button.setEnabled(True)

    def on_models_failed(self, error):
        self.status_label.setText(f"Unable to prepare RAG: {error}")

    def handle_send(self):
        query = self.input_box.text()
        if not query:
            return
        self.chat_display.append(f"You: {query}")
        try:
            answer = ask_rag(query, self.history)
        except Exception as error:
            answer = f"RAG request failed: {error}"
        self.chat_display.append(f"Bot: {answer}\n")
        self.history.append({"user": query, "assistant": answer})
        self.input_box.clear()


app = QApplication(sys.argv)
window = ChatWindow()
window.show()
sys.exit(app.exec())
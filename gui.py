import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
                             QCheckBox, QTextEdit, QFileDialog, QGroupBox, QGridLayout)
from PyQt6.QtCore import QProcess, Qt

class DubbingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Video Dubbing Pipeline - GUI")
        self.resize(1000, 800)
        self.apply_dark_theme()
        
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        
        self.initUI()
        
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #89b4fa;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #a6adc8;
            }
            QTextEdit {
                background-color: #11111b;
                border: 1px solid #45475a;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #89b4fa;
            }
        """)

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- SETTINGS GROUP ---
        settings_group = QGroupBox("Pipeline Ayarları")
        grid = QGridLayout()
        settings_group.setLayout(grid)
        
        # Row 0: Video Input
        self.video_input = QLineEdit("film.mp4")
        self.video_input.textChanged.connect(self.update_cli)
        btn_browse = QPushButton("Dosya Seç")
        btn_browse.clicked.connect(self.browse_file)
        grid.addWidget(QLabel("Video Dosyası:"), 0, 0)
        grid.addWidget(self.video_input, 0, 1, 1, 2)
        grid.addWidget(btn_browse, 0, 3)
        
        # Row 1: Languages
        self.tgt_lang_input = QLineEdit("Turkish")
        self.tgt_lang_input.textChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Hedef Dil (Ad):"), 1, 0)
        grid.addWidget(self.tgt_lang_input, 1, 1)
        
        self.tgt_lang_id_input = QLineEdit("tr")
        self.tgt_lang_id_input.textChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Hedef Dil (Kısa):"), 1, 2)
        grid.addWidget(self.tgt_lang_id_input, 1, 3)
        
        # Row 2: API Settings
        self.api_combo = QComboBox()
        self.api_combo.addItems(["lm_studio", "deepl"])
        self.api_combo.currentTextChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Çeviri API:"), 2, 0)
        grid.addWidget(self.api_combo, 2, 1)
        
        self.lm_url_input = QLineEdit("http://localhost:1234/v1")
        self.lm_url_input.textChanged.connect(self.update_cli)
        grid.addWidget(QLabel("LM Studio URL:"), 2, 2)
        grid.addWidget(self.lm_url_input, 2, 3)
        
        # Row 3: Audio Tweaks
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 3.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.25)
        self.speed_spin.valueChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Max Hızlandırma:"), 3, 0)
        grid.addWidget(self.speed_spin, 3, 1)
        
        self.vocal_vol_spin = QDoubleSpinBox()
        self.vocal_vol_spin.setRange(0.1, 5.0)
        self.vocal_vol_spin.setSingleStep(0.1)
        self.vocal_vol_spin.setValue(0.9)
        self.vocal_vol_spin.valueChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Vokal Sesi:"), 3, 2)
        grid.addWidget(self.vocal_vol_spin, 3, 3)
        
        # Row 4: Advanced
        self.shortening_spin = QDoubleSpinBox()
        self.shortening_spin.setRange(0.1, 1.0)
        self.shortening_spin.setSingleStep(0.05)
        self.shortening_spin.setValue(0.40)
        self.shortening_spin.valueChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Yavaşlatma Sınırı:"), 4, 0)
        grid.addWidget(self.shortening_spin, 4, 1)

        self.silence_input = QLineEdit("-40dB")
        self.silence_input.textChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Sessizlik Kırpma:"), 4, 2)
        grid.addWidget(self.silence_input, 4, 3)
        
        # Row 5: Flags & Extra
        self.bg_vol_spin = QDoubleSpinBox()
        self.bg_vol_spin.setRange(0.0, 5.0)
        self.bg_vol_spin.setSingleStep(0.1)
        self.bg_vol_spin.setValue(1.0)
        self.bg_vol_spin.valueChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Arkaplan Sesi:"), 5, 0)
        grid.addWidget(self.bg_vol_spin, 5, 1)
        
        self.chars_ps_spin = QDoubleSpinBox()
        self.chars_ps_spin.setRange(10.0, 35.0)
        self.chars_ps_spin.setSingleStep(1.0)
        self.chars_ps_spin.setValue(21.0)
        self.chars_ps_spin.valueChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Karakter/Sn:"), 5, 2)
        grid.addWidget(self.chars_ps_spin, 5, 3)

        self.context_spin = QSpinBox()
        self.context_spin.setRange(0, 10)
        self.context_spin.setValue(2)
        self.context_spin.setToolTip("Çeviri sırasında kaç önceki/sonraki segment bağlam olarak gönderilecek")
        self.context_spin.valueChanged.connect(self.update_cli)
        grid.addWidget(QLabel("Bağlam Penceresi:"), 5, 4)
        grid.addWidget(self.context_spin, 5, 5)
        
        self.chk_no_confirm = QCheckBox("Etkileşimli Soruları Atla (--no_confirm)")
        self.chk_no_confirm.stateChanged.connect(self.update_cli)
        grid.addWidget(self.chk_no_confirm, 6, 0, 1, 2)
        
        self.chk_keep_tmp = QCheckBox("Geçici Dosyaları Koru (--keep_tmp)")
        self.chk_keep_tmp.stateChanged.connect(self.update_cli)
        grid.addWidget(self.chk_keep_tmp, 6, 2, 1, 2)
        
        main_layout.addWidget(settings_group)
        
        # --- CLI PREVIEW ---
        cli_group = QGroupBox("Oluşturulan Komut (CLI Equivalent)")
        cli_layout = QVBoxLayout()
        cli_group.setLayout(cli_layout)
        
        self.cli_preview = QLineEdit()
        self.cli_preview.setReadOnly(True)
        self.cli_preview.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: monospace;")
        cli_layout.addWidget(self.cli_preview)
        
        main_layout.addWidget(cli_group)
        
        # --- START BUTTON ---
        self.btn_start = QPushButton("🚀 Pipeline'ı Başlat")
        self.btn_start.setStyleSheet("font-size: 15px; padding: 10px;")
        self.btn_start.clicked.connect(self.start_pipeline)
        main_layout.addWidget(self.btn_start)
        
        # --- LOG OUTPUT ---
        log_group = QGroupBox("Log Çıktısı")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        
        # Input section for interactive terminal
        terminal_input_layout = QHBoxLayout()
        self.terminal_input = QLineEdit()
        self.terminal_input.setPlaceholderText("Program soru sorarsa buraya cevabı yazıp Enter'a basın...")
        self.terminal_input.returnPressed.connect(self.send_input_to_process)
        
        btn_send = QPushButton("Gönder")
        btn_send.clicked.connect(self.send_input_to_process)
        
        terminal_input_layout.addWidget(self.terminal_input)
        terminal_input_layout.addWidget(btn_send)
        log_layout.addLayout(terminal_input_layout)
        
        main_layout.addWidget(log_group)
        
        self.update_cli()
        
    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Video Seç", "", "Video Files (*.mp4 *.mkv *.avi)")
        if fname:
            self.video_input.setText(fname)
            
    def generate_command_args(self):
        args = [
            "claude_main.py",
            "--video", self.video_input.text().strip(),
            "--target_language", self.tgt_lang_input.text().strip(),
            "--target_language_id", self.tgt_lang_id_input.text().strip(),
            "--translation_api", self.api_combo.currentText(),
            "--lm_studio_url", self.lm_url_input.text().strip(),
            "--max_speed_factor", str(self.speed_spin.value()),
            "--max_shortening_ratio", str(self.shortening_spin.value()),
            "--vocal_volume", str(self.vocal_vol_spin.value()),
            "--bg_volume", str(self.bg_vol_spin.value()),
            "--chars_per_second", str(self.chars_ps_spin.value()),
            "--context_size", str(self.context_spin.value()),
            f"--silence_threshold={self.silence_input.text().strip()}"
        ]
        if self.chk_no_confirm.isChecked():
            args.append("--no_confirm")
        if self.chk_keep_tmp.isChecked():
            args.append("--keep_tmp")
            
        return args

    def update_cli(self):
        args = self.generate_command_args()
        venv_python = "/mnt/depo_hdd/video_dubbing_Docker/.venv_2404/bin/python"
        python_display = venv_python if os.path.exists(venv_python) else sys.executable
        self.cli_preview.setText(python_display + " " + " ".join(args))
        
    def start_pipeline(self):
        self.log_output.clear()
        self.btn_start.setEnabled(False)
        self.btn_start.setText("⏳ Çalışıyor...")
        
        args = self.generate_command_args()
        
        # Force use the container's virtual environment python
        venv_python = "/mnt/depo_hdd/video_dubbing_Docker/.venv_2404/bin/python"
        python_exec = venv_python if os.path.exists(venv_python) else sys.executable
        
        self.append_log(f"> {python_exec} {' '.join(args)}\n")
        self.process.start(python_exec, args)
        
    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        self.append_log(data)
        
    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode('utf-8', errors='replace')
        self.append_log(data)
        
    def append_log(self, text):
        # Move cursor to end and insert
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_output.setTextCursor(cursor)
        self.log_output.insertPlainText(text)
        self.log_output.ensureCursorVisible()
        
    def send_input_to_process(self):
        if self.process.state() == QProcess.ProcessState.Running:
            text = self.terminal_input.text() + "\n"
            self.process.write(text.encode('utf-8'))
            self.append_log(f"\n[SİZ]: {text}")
            self.terminal_input.clear()
            
    def process_finished(self, exit_code, exit_status):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("🚀 Pipeline'ı Başlat")
        self.append_log(f"\n[SİSTEM] İşlem tamamlandı. (Çıkış kodu: {exit_code})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DubbingGUI()
    window.show()
    sys.exit(app.exec())

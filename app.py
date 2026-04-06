import json
import os
import zipfile
import urllib.request
import shutil
import threading
import subprocess
import queue
import torchaudio
from pathlib import Path

# torchaudio 2.x removed several APIs that pyannote still depends on.
# Patch them back in before pyannote is ever imported.
if not hasattr(torchaudio, "AudioMetaData"):
    from dataclasses import dataclass

    @dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = _AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# PyTorch 2.6 changed torch.load default to weights_only=True, breaking pyannote
# checkpoints that contain arbitrary globals (TorchVersion, etc.).
# Restore the pre-2.6 behaviour for calls that don't set weights_only explicitly.
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Lightning passes weights_only=None; PyTorch 2.6 treats None as True.
    # Only leave it alone if the caller explicitly requested True.
    if kwargs.get("weights_only") is not True:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# huggingface_hub 0.24+ removed use_auth_token in favour of token; pyannote 3.x still
# passes use_auth_token.  Wrap hf_hub_download before pyannote imports it.
import huggingface_hub as _hf_hub
_orig_hf_hub_download = _hf_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs.setdefault("token", kwargs.pop("use_auth_token"))
    return _orig_hf_hub_download(*args, **kwargs)
_hf_hub.hf_hub_download = _patched_hf_hub_download

from tkinter import (
    Tk,
    Frame,
    Label,
    Button,
    StringVar,
    ttk,
    filedialog,
    Text,
    END,
    Entry,
    Toplevel,
    messagebox,
)

APP_DIR = Path(__file__).parent
CONFIG_FILE = APP_DIR / "config.json"
MODELS_DIR = APP_DIR / "models"

DEFAULT_CONFIG = {
    "whisper": {
        "url": "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.4/whisper-cublas-11.8.0-bin-x64.zip",
        "version": "v1.8.4-cublas-11.8.0",
    },
    "llamacpp": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b8468/llama-b8468-bin-win-vulkan-x64.zip",
        "version": "b8468",
    },
    "models": {
        "llm": {
            "url": "https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true",
            "name": "Qwen3.5-9B-Q4_K_M.gguf",
            "display_name": "lmstudio-community/Qwen3.5-9B-GGUF",
        },
        "whisper": {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true",
            "name": "ggml-medium.bin",
            "display_name": "ggerganov/whisper.cpp",
        },
    },
    "hf_token": "",
}


class SpeakerNamingDialog:
    """
    Modal dialog for assigning human names to detected speakers.
    All name fields are shown at once; a single scrollable text box
    displays the full transcript with timestamps and placeholder IDs.

    Usage:
        dlg = SpeakerNamingDialog(parent, ["SPEAKER_00", "SPEAKER_01"], aligned=aligned)
        names = dlg.get_names()
        # {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
    """

    def __init__(self, parent, speakers: list, speaker_texts: dict = None, aligned: list = None):
        self._names: dict = {}
        self._entries: dict = {}
        self._build_ui(parent, speakers, aligned or [], speaker_texts or {})

    def _build_ui(self, parent, speakers, aligned: list, speaker_texts: dict):
        self.top = Toplevel(parent)
        self.top.title("Name the Speakers")
        self.top.resizable(True, True)
        self.top.grab_set()
        self.top.focus_set()

        Label(
            self.top,
            text="Enter a name for each speaker, then click Confirm.",
            font=("Arial", 10),
            pady=8,
        ).pack()

        # All speaker name fields visible at once
        names_frame = Frame(self.top, bd=1, relief="groove")
        names_frame.pack(fill="x", padx=10, pady=(0, 8))
        Label(names_frame, text="Speaker Names", font=("Arial", 10, "bold"), anchor="w", pady=4, padx=6).pack(fill="x")
        for i, speaker in enumerate(speakers):
            row = Frame(names_frame)
            row.pack(fill="x", padx=6, pady=2)
            Label(row, text=f"Speaker {i + 1} ({speaker}):", width=26, anchor="w").pack(side="left")
            var = StringVar(value=speaker)
            Entry(row, textvariable=var, width=28).pack(side="left", padx=4)
            self._entries[speaker] = var

        # Full transcript in a single scrollable text box
        Label(self.top, text="Full transcript:", font=("Arial", 10, "bold"), anchor="w", padx=10).pack(fill="x")
        text_frame = Frame(self.top)
        text_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        txt = Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, state="normal")
        txt.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=txt.yview)

        if aligned:
            lines = []
            for start_sec, speaker, text in aligned:
                minutes = int(start_sec) // 60
                seconds = int(start_sec) % 60
                lines.append(f"[{speaker} {minutes}:{seconds:02d}]: {text}")
            txt.insert(END, "\n".join(lines))
        elif speaker_texts:
            lines = []
            for spk, texts in speaker_texts.items():
                for t in texts:
                    lines.append(f"[{spk}]: {t}")
            txt.insert(END, "\n".join(lines))
        txt.config(state="disabled")

        btn_frame = Frame(self.top)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Confirm Names", command=self._on_confirm, width=14).pack()

        self.top.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        dw = max(self.top.winfo_reqwidth(), 600)
        dh = max(self.top.winfo_reqheight(), 520)
        self.top.geometry(f"{dw}x{dh}+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

        self.top.protocol("WM_DELETE_WINDOW", self._on_confirm)

    def _on_confirm(self):
        for speaker, var in self._entries.items():
            self._names[speaker] = var.get().strip() or speaker
        self.top.destroy()

    def get_names(self) -> dict:
        """Blocks (via nested event loop) until the dialog is closed."""
        self.top.wait_window()
        return self._names


class StatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LlamaWhisper Setup")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        self.config = self.load_or_create_config()
        self.setup_components()
        self.transcribe_button = None
        self.summarize_button = None
        self.create_status_screen()
        self.check_dependencies()

    def load_or_create_config(self):
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        else:
            with open(CONFIG_FILE, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG.copy()

    def setup_components(self):
        self.whisper_config = self.config["whisper"]
        self.llamacpp_config = self.config["llamacpp"]
        self.llm_config = self.config["models"]["llm"]
        self.whisper_model_config = self.config["models"]["whisper"]

        WHISPER_DIR = APP_DIR / "whisper"
        LLAMACPP_DIR = APP_DIR / "llamacpp"

        self.whisper_exe_dir = WHISPER_DIR / self.whisper_config["version"]
        self.llamacpp_exe_dir = LLAMACPP_DIR / self.llamacpp_config["version"]

        self.diarize_pipeline = None
        self.transcribe_process = None
        self.summarize_process = None
        self._transcribe_cancelled = False

    def create_status_screen(self):
        self.status_frame = Frame(self.root)
        self.status_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title = Label(
            self.status_frame, text="LlamaWhisper Setup", font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        info_frame = Frame(self.status_frame)
        info_frame.pack(fill="x", pady=5)
        Label(
            info_frame,
            text=f"whisper.cpp {self.whisper_config['version']}",
            font=("Arial", 9),
            fg="#666",
        ).pack(anchor="w")
        Label(
            info_frame,
            text=f"llama.cpp {self.llamacpp_config['version']}",
            font=("Arial", 9),
            fg="#666",
        ).pack(anchor="w")
        Label(
            info_frame,
            text=f"LLM: {self.llm_config['display_name']}",
            font=("Arial", 9),
            fg="#666",
        ).pack(anchor="w")
        Label(
            info_frame,
            text=f"Whisper model: {self.whisper_model_config['display_name']}",
            font=("Arial", 9),
            fg="#666",
        ).pack(anchor="w")

        Label(self.status_frame, text="", height=1).pack()

        self.whisper_label = self.create_status_row("whisper.cpp")
        self.llamacpp_label = self.create_status_row("llama.cpp")
        self.llm_label = self.create_status_row("LLM Model")
        self.whisper_model_label = self.create_status_row("Whisper Model")

        Label(self.status_frame, text="", height=1).pack()

        self.progress_var = StringVar(value="")
        self.progress_label = Label(
            self.status_frame, textvariable=self.progress_var, wraplength=620
        )
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(
            self.status_frame, mode="determinate", length=620
        )
        self.progress_bar.pack(pady=5)

        self.download_button = Button(
            self.status_frame,
            text="Download missing files",
            command=self.download_missing,
            state="normal",
        )
        self.download_button.pack(pady=10)

    def create_status_row(self, text):
        frame = Frame(self.status_frame)
        frame.pack(fill="x", pady=3)
        Label(frame, text=f"{text}:", width=18, anchor="w").pack(side="left")
        label = Label(frame, text="", font=("Arial", 10))
        label.pack(side="left")
        return label

    def update_status(self, label, status):
        if status == "Ready":
            label.config(text=status, fg="green")
        elif status == "Not found":
            label.config(text=status, fg="red")
        elif status == "Downloading":
            label.config(text=status, fg="orange")
        elif status == "Unpacking":
            label.config(text=status, fg="orange")
        else:
            label.config(text=status)

    def set_progress(self, percent, message=""):
        self.progress_bar["value"] = percent
        if message:
            self.progress_var.set(message)
        self.root.update()

    def check_whisper(self):
        if not self.whisper_exe_dir.exists():
            return False
        exe_path = self.whisper_exe_dir / "whisper-cli.exe"
        return exe_path.exists()

    def check_llamacpp(self):
        if not self.llamacpp_exe_dir.exists():
            return False
        exe_path = self.llamacpp_exe_dir / "llama-cli.exe"
        return exe_path.exists()

    def check_llm_model(self):
        if not MODELS_DIR.exists():
            return False
        model_path = MODELS_DIR / self.llm_config["name"]
        return model_path.exists()

    def check_whisper_model(self):
        if not MODELS_DIR.exists():
            return False
        model_path = MODELS_DIR / self.whisper_model_config["name"]
        return model_path.exists()

    def check_dependencies(self):
        whisper_ok = self.check_whisper()
        llamacpp_ok = self.check_llamacpp()
        llm_ok = self.check_llm_model()
        whisper_model_ok = self.check_whisper_model()

        self.update_status(self.whisper_label, "Ready" if whisper_ok else "Not found")
        self.update_status(self.llamacpp_label, "Ready" if llamacpp_ok else "Not found")
        self.update_status(self.llm_label, "Ready" if llm_ok else "Not found")
        self.update_status(
            self.whisper_model_label, "Ready" if whisper_model_ok else "Not found"
        )

        if whisper_ok and llamacpp_ok and llm_ok and whisper_model_ok:
            self.show_main_screen()
        else:
            self.download_button.config(state="normal")

    def download_missing(self):
        self.download_button.config(state="disabled")
        self.progress_bar["value"] = 0
        missing = []
        if not self.check_whisper():
            missing.append("whisper")
        if not self.check_llamacpp():
            missing.append("llamacpp")
        if not self.check_llm_model():
            missing.append("llm_model")
        if not self.check_whisper_model():
            missing.append("whisper_model")

        total = len(missing)

        for i, component in enumerate(missing):
            component_total = i + 1
            component_completed = component_total / total * 100

            if component == "whisper":
                self.update_status(self.whisper_label, "Downloading")
                self.set_progress(component_completed, f"Downloading whisper.cpp...")
                try:
                    self._download_whisper(self.set_progress)
                    self.update_status(self.whisper_label, "Unpacking")
                    self.set_progress(component_completed, "Unpacking whisper.cpp...")
                    self._unpack_whisper()
                    self.update_status(self.whisper_label, "Ready")
                except Exception as e:
                    self.update_status(self.whisper_label, "Error")
                    self.progress_var.set(f"Error: {e}")

            elif component == "llamacpp":
                self.update_status(self.llamacpp_label, "Downloading")
                self.set_progress(component_completed, f"Downloading llama.cpp...")
                try:
                    self._download_llamacpp(self.set_progress)
                    self.update_status(self.llamacpp_label, "Unpacking")
                    self.set_progress(component_completed, "Unpacking llama.cpp...")
                    self._unpack_llamacpp()
                    self.update_status(self.llamacpp_label, "Ready")
                except Exception as e:
                    self.update_status(self.llamacpp_label, "Error")
                    self.progress_var.set(f"Error: {e}")

            elif component == "llm_model":
                self.update_status(self.llm_label, "Downloading")
                self.set_progress(
                    component_completed,
                    f"Downloading {self.llm_config['display_name']}...",
                )
                try:
                    self._download_llm_model(self.set_progress)
                    self.update_status(self.llm_label, "Ready")
                except Exception as e:
                    self.update_status(self.llm_label, "Error")
                    self.progress_var.set(f"Error: {e}")

            elif component == "whisper_model":
                self.update_status(self.whisper_model_label, "Downloading")
                self.set_progress(
                    component_completed,
                    f"Downloading {self.whisper_model_config['display_name']}...",
                )
                try:
                    self._download_whisper_model(self.set_progress)
                    self.update_status(self.whisper_model_label, "Ready")
                except Exception as e:
                    self.update_status(self.whisper_model_label, "Error")
                    self.progress_var.set(f"Error: {e}")

        self.set_progress(100, "All downloads complete!")
        self.root.after(500, lambda: self.check_dependencies())

    def _download_whisper(self, progress_callback):
        zip_path = APP_DIR / "whisper" / "whisper-bin-x64.zip"
        (APP_DIR / "whisper").mkdir(parents=True, exist_ok=True)
        self._urlretrieve_with_progress(
            self.whisper_config["url"], zip_path, progress_callback
        )

    def _unpack_whisper(self):
        zip_path = APP_DIR / "whisper" / "whisper-bin-x64.zip"
        whisper_dir = APP_DIR / "whisper"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(whisper_dir)

        release_dir = whisper_dir / "Release"
        version_dir = whisper_dir / self.whisper_config["version"]

        if release_dir.exists():
            if version_dir.exists():
                shutil.rmtree(version_dir)
            release_dir.rename(version_dir)
        else:
            for item in whisper_dir.iterdir():
                if item.is_dir() and item.name != self.whisper_config["version"]:
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                    item.rename(version_dir)
                    break

        os.remove(zip_path)

    def _download_llamacpp(self, progress_callback):
        zip_path = APP_DIR / "llamacpp" / "llama-b8468-bin-win-vulkan-x64.zip"
        (APP_DIR / "llamacpp").mkdir(parents=True, exist_ok=True)
        self._urlretrieve_with_progress(
            self.llamacpp_config["url"], zip_path, progress_callback
        )

    def _unpack_llamacpp(self):
        zip_path = APP_DIR / "llamacpp" / "llama-b8468-bin-win-vulkan-x64.zip"
        llamacpp_dir = APP_DIR / "llamacpp"
        version_dir = llamacpp_dir / self.llamacpp_config["version"]
        version_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(llamacpp_dir)

        for item in list(llamacpp_dir.iterdir()):
            if item.is_file() and item.name != "llama-b8468-bin-win-vulkan-x64.zip":
                dest = version_dir / item.name
                shutil.move(str(item), str(dest))
            elif item.is_dir() and item.name != self.llamacpp_config["version"]:
                for subitem in item.iterdir():
                    dest = version_dir / subitem.name
                    if dest.exists():
                        if subitem.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(subitem), str(dest))
                item.rmdir()

        os.remove(zip_path)

    def _download_llm_model(self, progress_callback):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / self.llm_config["name"]
        self._urlretrieve_with_progress(
            self.llm_config["url"], model_path, progress_callback
        )

    def _download_whisper_model(self, progress_callback):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / self.whisper_model_config["name"]
        self._urlretrieve_with_progress(
            self.whisper_model_config["url"], model_path, progress_callback
        )

    def _urlretrieve_with_progress(self, url, filename, progress_callback):
        class ProgressHook:
            def __init__(self, progress_callback):
                self.downloaded = 0
                self.total = 0
                self.progress_callback = progress_callback

            def hook(self, block_num, block_size, total_size):
                self.downloaded += block_size
                if self.total == 0:
                    self.total = total_size
                if self.total > 0:
                    percent = min(100, (self.downloaded / self.total) * 100)
                    self.progress_callback(percent, f"Downloading... {percent:.1f}%")

        progress_hook = ProgressHook(progress_callback)
        urllib.request.urlretrieve(url, filename, progress_hook.hook)

    def show_main_screen(self):
        self.status_frame.pack_forget()
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        Label(self.main_frame, text="LlamaWhisper", font=("Arial", 24)).pack(pady=10)
        Label(self.main_frame, text="", height=1).pack()

        audio_frame = Frame(self.main_frame)
        audio_frame.pack(fill="x", pady=5)
        self.audio_file_var = StringVar(value="No audio file selected")
        self.audio_label = Label(
            audio_frame, textvariable=self.audio_file_var, anchor="w", width=50
        )
        self.audio_label.pack(side="left", expand=True, fill="x")
        Button(
            audio_frame, text="Choose audio", command=self.choose_audio_file, width=12
        ).pack(side="right", padx=5)

        Label(self.main_frame, text="", height=1).pack()

        lang_frame = Frame(self.main_frame)
        lang_frame.pack(fill="x", pady=5)
        Label(lang_frame, text="Language:", width=10).pack(side="left")
        self.language_var = StringVar(value="ru")
        self.language_entry = Entry(
            lang_frame, textvariable=self.language_var, width=10
        )
        self.language_entry.pack(side="left", padx=5)

        hf_frame = Frame(self.main_frame)
        hf_frame.pack(fill="x", pady=5)
        Label(hf_frame, text="HF Token:", width=10).pack(side="left")
        self.hf_token_var = StringVar(value=self.config.get("hf_token", ""))
        Entry(hf_frame, textvariable=self.hf_token_var, width=40, show="*").pack(
            side="left", padx=5
        )
        Button(hf_frame, text="Save", command=self._save_hf_token, width=6).pack(
            side="left"
        )

        self.transcribe_button = Button(
            self.main_frame,
            text="Diarize & Transcribe",
            command=self.transcribe,
            width=20,
        )
        self.transcribe_button.pack(pady=5)

        Label(self.main_frame, text="", height=1).pack()

        text_frame = Frame(self.main_frame)
        text_frame.pack(fill="x", pady=5)
        self.text_file_var = StringVar(value="No text file selected")
        self.text_label = Label(
            text_frame, textvariable=self.text_file_var, anchor="w", width=50
        )
        self.text_label.pack(side="left", expand=True, fill="x")
        Button(
            text_frame, text="Choose text", command=self.choose_text_file, width=12
        ).pack(side="right", padx=5)

        self.summarize_button = Button(
            self.main_frame, text="Summarize", command=self.summarize, width=15
        )
        self.summarize_button.pack(pady=5)

        Label(self.main_frame, text="", height=1).pack()

        self.output_text = Text(self.main_frame, height=15, wrap="word")
        self.output_text.pack(fill="both", expand=True, pady=10)
        self.output_text.tag_configure("info", foreground="blue")
        self.output_text.tag_configure("error", foreground="red")
        self.output_text.tag_configure("success", foreground="green")

    def _save_hf_token(self):
        self.config["hf_token"] = self.hf_token_var.get().strip()
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)
        self.log("HF token saved.", "success")

    def log(self, message, tag="info"):
        self.output_text.insert(END, message + "\n", tag)
        self.output_text.see(END)
        self.root.update()

    def read_output(self, process, output_queue):
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    output_queue.put(line.rstrip())
        finally:
            output_queue.put(None)

    def process_output_queue(self, output_queue, callback):
        try:
            while True:
                line = output_queue.get_nowait()
                if line is None:
                    callback()
                    break
                self.log(line.strip())
                self.root.update()
        except queue.Empty:
            pass
        self.root.after(100, lambda: self.process_output_queue(output_queue, callback))

    def choose_audio_file(self):
        filename = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.audio_file_var.set(filename)

    def choose_text_file(self):
        filename = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filename:
            self.text_file_var.set(filename)

    def transcribe(self):
        if self.transcribe_button.cget("text") == "Cancel":
            self.cancel_transcribe()
            return

        audio_file = self.audio_file_var.get()
        if audio_file == "No audio file selected":
            self.log("Please select an audio file first", "error")
            return

        hf_token = self.hf_token_var.get().strip()
        if not hf_token:
            self.log(
                "Hugging Face token is required for diarization. Enter it in the HF Token field and click Save.",
                "error",
            )
            return

        whisper_exe = self.whisper_exe_dir / "whisper-cli.exe"
        whisper_model = MODELS_DIR / self.whisper_model_config["name"]

        if not whisper_exe.exists():
            self.log(f"whisper-cli.exe not found at {whisper_exe}", "error")
            return
        if not whisper_model.exists():
            self.log(f"Whisper model not found at {whisper_model}", "error")
            return

        language = self.language_var.get().strip() or "ru"
        self._transcribe_cancelled = False
        self.transcribe_button.config(text="Cancel")

        t = threading.Thread(
            target=self._run_diarization_and_transcribe,
            args=(audio_file, hf_token, str(whisper_exe), str(whisper_model), language),
            daemon=True,
        )
        t.start()

    def _run_diarization_and_transcribe(
        self, audio_file, hf_token, whisper_exe, whisper_model, language
    ):
        def ui(fn):
            self.root.after(0, fn)

        def log(msg, tag="info"):
            ui(lambda m=msg, t=tag: self.log(m, t))

        try:
            # Stage 1: Diarization
            log("Loading diarization pipeline (first run may take a minute)...")
            diarize_segs = self.diarize_audio(audio_file, hf_token)

            if self._transcribe_cancelled:
                return

            speaker_count = len(set(s[2] for s in diarize_segs))
            log(f"Diarization complete: {speaker_count} speaker(s) detected.")

            # Stage 2: Whisper JSON transcription
            log("Running Whisper transcription...")
            json_path = self.run_whisper_json(
                audio_file, whisper_exe, whisper_model, language
            )

            if self._transcribe_cancelled:
                return

            if json_path is None:
                log("Whisper transcription failed.", "error")
                ui(lambda: self.transcribe_button.config(text="Diarize & Transcribe"))
                return

            # Stage 3: Parse whisper JSON
            whisper_segs = self.parse_whisper_json(json_path)
            log(f"Parsed {len(whisper_segs)} transcript segment(s).")

            # Stage 4: Align speakers to transcript segments
            aligned = self.align_speakers(whisper_segs, diarize_segs)

            # Stage 5: Speaker naming dialog (must run on main thread)
            speakers = sorted(set(item[1] for item in aligned))
            speaker_texts = {}
            for _, spk, txt in aligned:
                speaker_texts.setdefault(spk, []).append(txt)
            name_result = {}
            dialog_done = threading.Event()

            def show_dialog(spk_texts=speaker_texts, al=aligned):
                dlg = SpeakerNamingDialog(self.root, speakers, spk_texts, al)
                name_result.update(dlg.get_names())
                dialog_done.set()

            ui(show_dialog)
            dialog_done.wait()

            if self._transcribe_cancelled:
                return

            # Stage 6: Write stitched transcript
            transcript_path = self.write_transcript(audio_file, aligned, name_result)
            log(f"Transcript saved to {transcript_path}", "success")

            # Stage 7: Auto-populate text field and reset button
            def finish():
                self.text_file_var.set(transcript_path)
                self.transcribe_button.config(text="Diarize & Transcribe")
                self.log("Ready to summarize.", "success")

            ui(finish)

        except Exception as e:
            def report(err=e):
                self.log(f"Error: {err}", "error")
                self.transcribe_button.config(text="Diarize & Transcribe")

            ui(report)

    def diarize_audio(self, audio_file: str, hf_token: str) -> list:
        """
        Returns [(start_sec, end_sec, speaker_label), ...].

        Loads audio via soundfile + ffmpeg (bypasses torchcodec entirely,
        avoiding DLL issues on Windows).
        """
        import tempfile

        try:
            from static_ffmpeg import run as ffmpeg_run
            ffmpeg_exe, _ = ffmpeg_run.get_or_fetch_platform_executables_else_raise()
        except Exception as e:
            raise RuntimeError(
                f"Failed to locate bundled ffmpeg: {e}. Run: uv add static-ffmpeg"
            )

        try:
            import soundfile as sf
            import torch
            import numpy as np
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Run: uv sync")

        if self.diarize_pipeline is None:
            self.diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )

        # Formats soundfile can't read natively — convert to WAV first.
        # This also completely avoids torchcodec's ffmpeg DLL requirement on Windows.
        needs_conversion = Path(audio_file).suffix.lower() in (
            ".mp3", ".m4a", ".aac", ".ogg", ".opus"
        )
        tmp_wav = None
        try:
            if needs_conversion:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_wav = tmp.name
                tmp.close()
                subprocess.run(
                    [ffmpeg_exe, "-i", audio_file, "-ar", "16000", "-ac", "1", "-y", tmp_wav],
                    check=True,
                    capture_output=True,
                )
                load_path = tmp_wav
            else:
                load_path = audio_file

            # Load with soundfile — no torchcodec, no DLL dependencies
            data, sr = sf.read(load_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)  # [channels, samples]
        finally:
            if tmp_wav:
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass

        # Pass pre-loaded waveform so pyannote skips its own audio I/O entirely.
        # pyannote 3.x returns the Annotation directly; 4.x wraps it in a
        # DiarizeOutput dataclass with a .speaker_diarization attribute.
        output = self.diarize_pipeline({"waveform": waveform, "sample_rate": sr})
        annotation = getattr(output, "speaker_diarization", output)
        return [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

    def run_whisper_json(
        self, audio_file: str, whisper_exe: str, whisper_model: str, language: str
    ):
        """
        Runs whisper-cli with -oj flag. Blocks until complete.
        Returns path to the output JSON file, or None on failure.
        """
        cmd = [
            whisper_exe,
            "-m", whisper_model,
            "-l", language,
            "-f", audio_file,
            "-oj",
            "-pp",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.transcribe_process = process

        for line in iter(process.stdout.readline, ""):
            stripped = line.strip()
            if stripped:
                self.root.after(0, lambda l=stripped: self.log(l))

        process.wait()

        if process.returncode != 0 and not self._transcribe_cancelled:
            return None

        json_path = audio_file + ".json"
        return json_path if Path(json_path).exists() else None

    def parse_whisper_json(self, json_path: str) -> list:
        """Returns [(start_sec, end_sec, text), ...]."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            (
                entry["offsets"]["from"] / 1000.0,
                entry["offsets"]["to"] / 1000.0,
                entry["text"].strip(),
            )
            for entry in data.get("transcription", [])
            if entry.get("text", "").strip()
        ]

    def align_speakers(self, whisper_segs: list, diarize_segs: list) -> list:
        """
        Assigns each whisper segment to the diarization speaker with max overlap.
        Returns [(start_sec, speaker_label, text), ...] sorted by start_sec.
        """
        result = []
        for w_start, w_end, text in whisper_segs:
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0
            for d_start, d_end, speaker in diarize_segs:
                overlap = max(0.0, min(w_end, d_end) - max(w_start, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
            result.append((w_start, best_speaker, text))

        return sorted(result, key=lambda x: x[0])

    def write_transcript(self, audio_file: str, aligned: list, name_map: dict) -> str:
        """
        Writes aligned segments to {audio_file}_transcript.txt.
        Format: [Name M:SS]: text
        Returns the output path.
        """
        output_path = str(Path(audio_file).with_suffix("")) + "_transcript.txt"
        lines = []
        for start_sec, speaker, text in aligned:
            name = name_map.get(speaker, speaker)
            minutes = int(start_sec) // 60
            seconds = int(start_sec) % 60
            lines.append(f"[{name} {minutes}:{seconds:02d}]: {text}")

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def cancel_transcribe(self):
        self._transcribe_cancelled = True
        if self.transcribe_process:
            self.log("Cancelling transcription...")
            try:
                self.transcribe_process.terminate()
                self.transcribe_process.wait(timeout=5)
            except Exception:
                try:
                    self.transcribe_process.kill()
                except Exception:
                    pass
            self.transcribe_process = None
        self.transcribe_button.config(text="Diarize & Transcribe")
        self.log("Transcription cancelled", "error")

    def summarize(self):
        if self.summarize_button.cget("text") == "Cancel":
            self.cancel_summarize()
            return

        text_file = self.text_file_var.get()
        if text_file == "No text file selected":
            self.log("Please select a text file first", "error")
            return

        llama_exe = self.llamacpp_exe_dir / "llama-cli.exe"
        llm_model = MODELS_DIR / self.llm_config["name"]

        if not llama_exe.exists():
            self.log(f"llama-cli.exe not found at {llama_exe}", "error")
            return
        if not llm_model.exists():
            self.log(f"LLM model not found at {llm_model}", "error")
            return

        prompt_file = APP_DIR / "system-summary.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(
                "Кратко изложи содержание этой стенограммы. "
                "Стенограмма содержит реплики нескольких участников в формате "
                "[Имя ЧЧ:ММ]: текст. Учитывай, кто что говорил.\n"
            )

        self.log("Starting summarization")
        cmd = [
            str(llama_exe),
            "-m",
            str(llm_model),
            "--temp",
            "0",
            "--simple-io",
            "-st",
            "--reasoning",
            "off",
            "-sysf",
            str(prompt_file),
            "-f",
            text_file,
        ]

        summary_path = APP_DIR / "summary.txt"
        output_lines = []

        def completion_callback():
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            self.log(f"Summary saved to {summary_path}", "success")
            self.summarize_button.config(text="Summarize")

        try:
            self.log(f"Running llama.cpp with prompt file")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self.summarize_process = process
            output_queue = queue.Queue()
            read_thread = threading.Thread(
                target=self.read_output_with_capture,
                args=(process, output_queue, output_lines),
                daemon=True,
            )
            read_thread.start()
            self.summarize_button.config(text="Cancel")
            self.root.after(
                100,
                lambda: self.process_output_queue(output_queue, completion_callback),
            )
        except Exception as e:
            self.summarize_button.config(text="Summarize")
            self.log(f"Error: {e}", "error")

    def cancel_summarize(self):
        if self.summarize_process:
            self.log("Cancelling summarization...")
            try:
                self.summarize_process.terminate()
                self.summarize_process.wait(timeout=5)
            except Exception:
                try:
                    self.summarize_process.kill()
                except Exception:
                    pass
            self.summarize_process = None
            self.summarize_button.config(text="Summarize")
            self.log("Summarization cancelled", "error")

    def read_output_with_capture(self, process, output_queue, output_lines):
        capturing = False
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    stripped = line.rstrip()
                    if ">" in stripped:
                        capturing = True
                    if capturing:
                        if "llama_memory_breakdown_print" in stripped:
                            break
                        output_queue.put(stripped)
                        output_lines.append(stripped)
        finally:
            output_queue.put(None)


def main():
    root = Tk()
    app = StatusApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

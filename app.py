import json
import os
import sys
import platform
import zipfile
import tarfile
import urllib.request
import shutil
import threading
import subprocess
import queue
from pathlib import Path
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
)

APP_DIR = Path(__file__).parent
CONFIG_FILE = APP_DIR / "config.json"
MODELS_DIR = APP_DIR / "models"

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
MACOS_ARCH = platform.machine()  # "arm64" or "x86_64"


def _exe(name):
    """Return platform-appropriate binary name."""
    return name + (".exe" if IS_WINDOWS else "")


def _make_default_config():
    if IS_MACOS:
        macos_arch = "arm64" if MACOS_ARCH == "arm64" else "x64"
        return {
            "whisper": {
                "url": "",
                "version": "v1.8.4",
            },
            "llamacpp": {
                "url": f"https://github.com/ggml-org/llama.cpp/releases/download/b8468/llama-b8468-bin-macos-{macos_arch}.tar.gz",
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
        }
    else:
        return {
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
        }


DEFAULT_CONFIG = _make_default_config()


class StatusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LlamaWhisper Setup")
        self.root.geometry("700x550")
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
        elif status == "Building":
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
        exe_path = self.whisper_exe_dir / _exe("whisper-cli")
        return exe_path.exists()

    def check_llamacpp(self):
        if not self.llamacpp_exe_dir.exists():
            return False
        exe_path = self.llamacpp_exe_dir / _exe("llama-cli")
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
                if IS_MACOS:
                    self.update_status(self.whisper_label, "Building")
                    self.set_progress(
                        component_completed, "Building whisper.cpp from source..."
                    )
                    try:
                        self._build_whisper_macos()
                        self.update_status(self.whisper_label, "Ready")
                    except Exception as e:
                        self.update_status(self.whisper_label, "Error")
                        self.progress_var.set(f"Error: {e}")
                else:
                    self.update_status(self.whisper_label, "Downloading")
                    self.set_progress(
                        component_completed, f"Downloading whisper.cpp..."
                    )
                    try:
                        self._download_whisper(self.set_progress)
                        self.update_status(self.whisper_label, "Unpacking")
                        self.set_progress(
                            component_completed, "Unpacking whisper.cpp..."
                        )
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

    # ------------------------------------------------------------------ #
    #  whisper.cpp — Windows: download zip  /  macOS: build from source   #
    # ------------------------------------------------------------------ #

    def _download_whisper(self, progress_callback):
        """Windows only: download the prebuilt whisper.cpp zip."""
        zip_path = APP_DIR / "whisper" / "whisper-bin-x64.zip"
        (APP_DIR / "whisper").mkdir(parents=True, exist_ok=True)
        self._urlretrieve_with_progress(
            self.whisper_config["url"], zip_path, progress_callback
        )

    def _unpack_whisper(self):
        """Windows only: unpack the downloaded whisper.cpp zip."""
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

    def _check_macos_prerequisites(self):
        """Raise RuntimeError if cmake or git are not available on macOS."""
        missing = []
        for tool in ("cmake", "git"):
            try:
                subprocess.run(
                    [tool, "--version"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(tool)
        if missing:
            tools_str = " and ".join(missing)
            raise RuntimeError(
                f"{tools_str} not found. "
                "Please install Xcode Command Line Tools (`xcode-select --install`) "
                "or Homebrew with `brew install cmake git`, then try again."
            )

    def _build_whisper_macos(self):
        """
        macOS only: download whisper.cpp source, build whisper-cli with cmake,
        install the binary to the version directory, and stream build output
        to the progress label so the user can see what is happening.
        """
        self._check_macos_prerequisites()

        version = self.whisper_config["version"]  # e.g. "v1.8.4"
        src_url = f"https://github.com/ggml-org/whisper.cpp/archive/refs/tags/{version}.tar.gz"
        whisper_dir = APP_DIR / "whisper"
        src_dir = whisper_dir / "_src"
        version_dir = whisper_dir / version

        whisper_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Download source tarball ---
        src_tarball = whisper_dir / f"whisper-{version}-src.tar.gz"
        self.set_progress(5, f"Downloading whisper.cpp {version} source...")
        self._urlretrieve_with_progress(src_url, src_tarball, self.set_progress)

        # --- 2. Extract source ---
        self.set_progress(15, "Extracting whisper.cpp source...")
        if src_dir.exists():
            shutil.rmtree(src_dir)
        src_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src_tarball, "r:gz") as tf:
            tf.extractall(src_dir)
        os.remove(src_tarball)

        # The tarball extracts to a single top-level directory like
        # whisper.cpp-1.8.4/ (without the leading "v")
        inner_name = f"whisper.cpp-{version.lstrip('v')}"
        build_root = src_dir / inner_name
        if not build_root.exists():
            # Fallback: find whatever single directory was extracted
            subdirs = [p for p in src_dir.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                build_root = subdirs[0]
            else:
                raise RuntimeError(
                    f"Unexpected source layout after extraction: {list(src_dir.iterdir())}"
                )

        build_dir = build_root / "build"

        # --- 3. cmake configure ---
        self.set_progress(20, "Configuring whisper.cpp (cmake)...")
        self._run_subprocess_logged(
            [
                "cmake",
                "-B",
                str(build_dir),
                "-S",
                str(build_root),
                "-DGGML_METAL=ON",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=str(build_root),
            progress_start=20,
            progress_end=30,
        )

        # --- 4. cmake build ---
        cpu_count = str(os.cpu_count() or 4)
        self.set_progress(30, "Building whisper-cli (this may take a few minutes)...")
        self._run_subprocess_logged(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "whisper-cli",
                "-j",
                cpu_count,
            ],
            cwd=str(build_root),
            progress_start=30,
            progress_end=90,
        )

        # --- 5. Install binary ---
        self.set_progress(90, "Installing whisper-cli...")
        built_binary = build_dir / "bin" / "whisper-cli"
        if not built_binary.exists():
            # Some cmake configurations put it directly in build/
            built_binary = build_dir / "whisper-cli"
        if not built_binary.exists():
            raise RuntimeError(
                f"Build completed but whisper-cli not found. "
                f"Searched: {build_dir / 'bin' / 'whisper-cli'} and {build_dir / 'whisper-cli'}"
            )

        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / "whisper-cli"
        shutil.copy2(str(built_binary), str(dest))
        os.chmod(dest, 0o755)

        # --- 6. Cleanup source ---
        shutil.rmtree(src_dir)
        self.set_progress(95, "whisper-cli built successfully.")

    def _run_subprocess_logged(self, cmd, cwd, progress_start, progress_end):
        """
        Run a subprocess, stream its combined stdout/stderr to the progress
        label, and interpolate progress_bar between progress_start and
        progress_end.  Raises RuntimeError on non-zero exit.
        """
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines_seen = 0
        # We don't know total lines, so we drift the progress bar slowly
        # toward progress_end as output accumulates.
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                lines_seen += 1
                # Asymptotically approach progress_end
                span = progress_end - progress_start
                current = progress_end - span / (1 + lines_seen / 10)
                self.set_progress(current, line[:120])
        process.stdout.close()
        rc = process.wait()
        if rc != 0:
            raise RuntimeError(
                f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}"
            )

    # ------------------------------------------------------------------ #
    #  llama.cpp — Windows: zip  /  macOS: tar.gz                         #
    # ------------------------------------------------------------------ #

    def _llamacpp_archive_name(self):
        """Return the expected archive filename for the current platform."""
        url = self.llamacpp_config["url"]
        return Path(url).name  # e.g. "llama-b8468-bin-win-vulkan-x64.zip"
        #   or "llama-b8468-bin-macos-arm64.tar.gz"

    def _download_llamacpp(self, progress_callback):
        archive_name = self._llamacpp_archive_name()
        archive_path = APP_DIR / "llamacpp" / archive_name
        (APP_DIR / "llamacpp").mkdir(parents=True, exist_ok=True)
        self._urlretrieve_with_progress(
            self.llamacpp_config["url"], archive_path, progress_callback
        )

    def _unpack_llamacpp(self):
        archive_name = self._llamacpp_archive_name()
        archive_path = APP_DIR / "llamacpp" / archive_name
        llamacpp_dir = APP_DIR / "llamacpp"
        version_dir = llamacpp_dir / self.llamacpp_config["version"]
        version_dir.mkdir(parents=True, exist_ok=True)

        if IS_MACOS:
            self._unpack_llamacpp_targz(
                archive_path, llamacpp_dir, version_dir, archive_name
            )
        else:
            self._unpack_llamacpp_zip(
                archive_path, llamacpp_dir, version_dir, archive_name
            )

        os.remove(archive_path)

    def _unpack_llamacpp_zip(self, zip_path, llamacpp_dir, version_dir, zip_name):
        """Windows: extract zip, flatten everything into version_dir."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(llamacpp_dir)

        for item in list(llamacpp_dir.iterdir()):
            if item.is_file() and item.name != zip_name:
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

    def _unpack_llamacpp_targz(self, tar_path, llamacpp_dir, version_dir, tar_name):
        """
        macOS: extract tar.gz.  The llama.cpp macOS release tarballs contain
        all binaries flat at the top level (no wrapper directory).
        After extraction, chmod the CLI binary executable.
        """
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(llamacpp_dir)

        # Move any files/dirs that landed in llamacpp_dir (not the archive
        # itself and not the version dir) into version_dir.
        for item in list(llamacpp_dir.iterdir()):
            if item.name == tar_name or item.name == self.llamacpp_config["version"]:
                continue
            dest = version_dir / item.name
            if dest.exists():
                if item.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        # Ensure the CLI binary is executable
        cli = version_dir / "llama-cli"
        if cli.exists():
            os.chmod(cli, 0o755)

    # ------------------------------------------------------------------ #
    #  Model downloads (platform-neutral)                                  #
    # ------------------------------------------------------------------ #

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

        self.transcribe_button = Button(
            self.main_frame, text="Transcribe", command=self.transcribe, width=15
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

        whisper_exe = self.whisper_exe_dir / _exe("whisper-cli")
        whisper_model = MODELS_DIR / self.whisper_model_config["name"]

        if not whisper_exe.exists():
            self.log(f"{_exe('whisper-cli')} not found at {whisper_exe}", "error")
            return
        if not whisper_model.exists():
            self.log(f"Whisper model not found at {whisper_model}", "error")
            return

        self.log(f"Starting transcription of {audio_file}")
        language = self.language_var.get().strip()
        if not language:
            language = "ru"
        cmd = [
            str(whisper_exe),
            "-m",
            str(whisper_model),
            "-l",
            language,
            "-f",
            audio_file,
            "-otxt",
            "-pp",
        ]

        def completion_callback():
            self.transcribe_button.config(text="Transcribe")
            self.log("Transcription complete", "success")

        try:
            self.log(f"Running: {' '.join(cmd)}")
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
            output_queue = queue.Queue()
            read_thread = threading.Thread(
                target=self.read_output, args=(process, output_queue), daemon=True
            )
            read_thread.start()
            self.transcribe_button.config(text="Cancel")
            self.root.after(
                100,
                lambda: self.process_output_queue(output_queue, completion_callback),
            )
        except Exception as e:
            self.transcribe_button.config(text="Transcribe")
            self.log(f"Error: {e}", "error")

    def cancel_transcribe(self):
        if self.transcribe_process:
            self.log("Cancelling transcription...")
            try:
                self.transcribe_process.terminate()
                self.transcribe_process.wait(timeout=5)
            except:
                try:
                    self.transcribe_process.kill()
                except:
                    pass
            self.transcribe_process = None
            self.transcribe_button.config(text="Transcribe")
            self.log("Transcription cancelled", "error")

    def summarize(self):
        if self.summarize_button.cget("text") == "Cancel":
            self.cancel_summarize()
            return

        text_file = self.text_file_var.get()
        if text_file == "No text file selected":
            self.log("Please select a text file first", "error")
            return

        llama_exe = self.llamacpp_exe_dir / _exe("llama-cli")
        llm_model = MODELS_DIR / self.llm_config["name"]

        if not llama_exe.exists():
            self.log(f"{_exe('llama-cli')} not found at {llama_exe}", "error")
            return
        if not llm_model.exists():
            self.log(f"LLM model not found at {llm_model}", "error")
            return

        prompt_file = APP_DIR / "system-summary.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(f"Кратко изложи содержание этой стенограммы:\n")

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
            except:
                try:
                    self.summarize_process.kill()
                except:
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

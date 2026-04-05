# LlamaWhisper

Desktop application for extracting conversations from audio files and formatting the text into organized reports using local AI models. Vibecoded by Qwen3.5-122B-A10B (Q6_K_L).

## Features

- **Speaker Diarization**: Automatically detect who spoke when using pyannote.audio
- **Audio Transcription**: Convert speech to text using whisper.cpp with timestamps
- **Multi-speaker Transcript**: Name each detected speaker and get a stitched transcript with `[Name M:SS]: text` format
- **Text Summarization**: Generate concise summaries of transcripts using Qwen3.5 LLM
- **Local Processing**: All AI models run locally - no cloud dependencies (except HuggingFace for the diarization model download)
- **Progress Tracking**: Real-time download and processing progress indicators
- **Cancel Support**: Stop transcription or summarization at any time

## Requirements

- Windows 10/11
- NVIDIA GPU with CUDA support (for accelerated inference)
- Python 3.8+
- ~20GB free disk space for models and binaries (additional ~5GB for PyTorch/pyannote)
- A free HuggingFace account (required for the diarization model)

## HuggingFace Account Setup

Speaker diarization uses [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), which requires a free HuggingFace account and a one-time model agreement:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the model terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Paste the token into the **HF Token** field in the app and click **Save**

The token is stored locally in `config.json` and never sent anywhere other than HuggingFace for model download.

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
uv sync
```

3. Run the application:
```bash
uv run app.py
```

The application will automatically download the required binaries and models on first run:
- **whisper.cpp** (v1.8.4) - for audio transcription
- **llama.cpp** (b8468) - for LLM inference
- **Qwen3.5-9B-Q4_K_M.gguf** - LLM model for summarization (~5GB)
- **ggml-medium.bin** - Whisper model for transcription (~700MB)

The pyannote diarization model (~1GB) is downloaded automatically on first transcription.

## Usage

### Diarization & Transcription

1. Click **Choose audio** and select an audio file (supports `.wav`, `.mp3`, `.flac`, `.m4a`)
2. Set the language code (default: `ru`)
3. Enter your HuggingFace token in the **HF Token** field and click **Save** (first time only)
4. Click **Diarize & Transcribe** to start
   - The app first runs speaker diarization to detect who spoke when
   - Then runs Whisper to transcribe the audio with timestamps
   - A dialog appears with one tab per detected speaker
5. In the dialog, enter a display name for each speaker and click **Confirm Names**
6. The stitched transcript is saved as `{audio_filename}_transcript.txt` alongside the audio file and automatically loaded into the Summarize input

### Summarization

1. The transcript file is auto-loaded after transcription, or click **Choose text** to select one manually
2. Click **Summarize** to generate a summary
3. The summary will be saved to `summary.txt` in the application directory
4. Click **Cancel** to stop summarization

### Transcript Format

The stitched transcript uses the format:
```
[Alice 0:00]: Hello, how are you today?
[Bob 0:08]: I'm doing well, thanks for asking.
[Alice 0:15]: Great to hear. Let's get started.
```

## Configuration

Edit `config.json` to customize:
- Download URLs for binaries
- Model versions
- Model file locations
- HuggingFace token (`hf_token`)

## Project Structure

```
llamawhisper/
├── app.py              # Main application
├── config.json         # Configuration file (includes hf_token)
├── system-summary.txt  # System prompt for summarization
├── models/             # Downloaded AI models
│   ├── ggml-medium.bin
│   └── Qwen3.5-9B-Q4_K_M.gguf
├── whisper/            # whisper.cpp binaries
└── llamacpp/           # llama.cpp binaries
```

## Technologies

- **pyannote.audio**: Speaker diarization (who spoke when)
- **whisper.cpp**: Efficient speech recognition in C/C++
- **llama.cpp**: LLM inference in pure C/C++
- **Qwen3.5**: Large language model for text summarization
- **Tkinter**: Python GUI toolkit

## License

This project uses open-source components:
- whisper.cpp (MIT License)
- llama.cpp (MIT License)
- Qwen3.5 (Apache 2.0 License)
- pyannote.audio (MIT License)

---

# LlamaWhisper

Десктопное приложение для извлечения разговоров из звуковых файлов и форматирования текста в упорядоченные отчеты с использованием локальных ИИ-моделей. Создано с помощью языковой модели Qwen3.5-122B-A10B (Q6_K_L).

## Возможности

- **Диаризация речи**: Автоматическое определение того, кто и когда говорил, с помощью pyannote.audio
- **Расшифровка аудио**: Преобразование речи в текст с помощью whisper.cpp с временными метками
- **Многоголосой транскрипт**: Назовите каждого участника и получите сшитый транскрипт в формате `[Имя М:СС]: текст`
- **Резюмирование текста**: Генерация кратких саммари транскрипций с помощью LLM Qwen3.5
- **Локальная обработка**: Все ИИ-модели работают локально — без облачных зависимостей (кроме HuggingFace для загрузки модели диаризации)
- **Отслеживание прогресса**: Индикаторы загрузки и обработки в реальном времени
- **Возможность отмены**: Прервите расшифровку или резюмирование в любой момент

## Требования

- Windows 10/11
- GPU NVIDIA с поддержкой CUDA (для ускорения инференса)
- Python 3.8+
- ~20 ГБ свободного места на диске для моделей и бинарников (дополнительно ~5 ГБ для PyTorch/pyannote)
- Бесплатный аккаунт HuggingFace (необходим для модели диаризации)

## Настройка аккаунта HuggingFace

Диаризация речи использует [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), для которой необходим бесплатный аккаунт HuggingFace и однократное принятие условий использования модели:

1. Создайте бесплатный аккаунт на [huggingface.co](https://huggingface.co)
2. Примите условия использования модели на странице [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Примите условия использования модели на странице [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Создайте токен доступа на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Вставьте токен в поле **HF Token** в приложении и нажмите **Save**

Токен хранится локально в `config.json` и никуда не передаётся, кроме HuggingFace для загрузки модели.

## Установка

1. Клонируйте или скачайте этот репозиторий

2. Установите зависимости:
```bash
uv sync
```

3. Запустите приложение:
```bash
uv run app.py
```

Приложение автоматически загрузит необходимые бинарники и модели при первом запуске:
- **whisper.cpp** (v1.8.4) — для расшифровки аудио
- **llama.cpp** (b8468) — для инференса LLM
- **Qwen3.5-9B-Q4_K_M.gguf** — модель LLM для резюмирования (~5 ГБ)
- **ggml-medium.bin** — модель Whisper для расшифровки (~700 МБ)

Модель диаризации pyannote (~1 ГБ) загружается автоматически при первой расшифровке.

## Использование

### Диаризация и расшифровка

1. Нажмите **Choose audio** и выберите аудиофайл (поддерживаются `.wav`, `.mp3`, `.flac`, `.m4a`)
2. Укажите код языка (по умолчанию: `ru`)
3. Введите токен HuggingFace в поле **HF Token** и нажмите **Save** (только при первом использовании)
4. Нажмите **Diarize & Transcribe** для запуска
   - Приложение сначала выполняет диаризацию речи для определения, кто и когда говорил
   - Затем запускает Whisper для расшифровки аудио с временными метками
   - Откроется диалоговое окно с вкладкой для каждого обнаруженного участника
5. В диалоговом окне введите имя для каждого участника и нажмите **Confirm Names**
6. Сшитый транскрипт сохраняется как `{имя_аудиофайла}_transcript.txt` рядом с аудиофайлом и автоматически загружается в поле для резюмирования

### Резюмирование

1. Файл транскрипта загружается автоматически после расшифровки, или нажмите **Choose text** для выбора вручную
2. Нажмите **Summarize** для генерации саммари
3. Саммари будет сохранено в `summary.txt` в директории приложения
4. Нажмите **Cancel** для остановки резюмирования

### Формат транскрипта

Сшитый транскрипт использует формат:
```
[Алиса 0:00]: Привет, как дела?
[Боб 0:08]: Всё хорошо, спасибо.
[Алиса 0:15]: Отлично. Давай начнём.
```

## Конфигурация

Отредактируйте `config.json` для настройки:
- URL-адресов загрузки бинарников
- Версий моделей
- Расположения файлов моделей
- Токена HuggingFace (`hf_token`)

## Структура проекта

```
llamawhisper/
├── app.py              # Основное приложение
├── config.json         # Файл конфигурации (включает hf_token)
├── system-summary.txt  # Системный промпт для резюмирования
├── models/             # Скачанные ИИ-модели
│   ├── ggml-medium.bin
│   └── Qwen3.5-9B-Q4_K_M.gguf
├── whisper/            # Бинарники whisper.cpp
└── llamacpp/           # Бинарники llama.cpp
```

## Технологии

- **pyannote.audio**: Диаризация речи (кто и когда говорил)
- **whisper.cpp**: Эффективное распознавание речи на C/C++
- **llama.cpp**: Инференс LLM на чистом C/C++
- **Qwen3.5**: Большая языковая модель для резюмирования текста
- **Tkinter**: GUI-тулкит для Python

## Лицензия

Проект использует компоненты с открытым исходным кодом:
- whisper.cpp (MIT License)
- llama.cpp (MIT License)
- Qwen3.5 (Apache 2.0 License)
- pyannote.audio (MIT License)

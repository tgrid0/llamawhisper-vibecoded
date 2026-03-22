# LlamaWhisper

Desktop application for extracting conversations from audio files and formatting the text into organized reports using local AI models. Vibecoded by Qwen3.5-122B-A10B (Q6_K_L).

## Features

- **Audio Transcription**: Convert speech to text using whisper.cpp with Russian language support
- **Text Summarization**: Generate concise summaries of transcripts using Qwen3.5 LLM
- **Local Processing**: All AI models run locally - no cloud dependencies
- **Progress Tracking**: Real-time download and processing progress indicators
- **Cancel Support**: Stop transcription or summarization at any time

## Requirements

- Windows 10/11
- NVIDIA GPU with CUDA support (for accelerated inference)
- Python 3.8+
- ~15GB free disk space for models and binaries

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

## Usage

### Transcription

1. Click "Choose audio" and select an audio file (supports `.wav`, `.mp3`, `.flac`, `.m4a`)
2. Click "Transcribe" to start transcription
3. The transcript will be saved as a `.txt` file alongside the audio file
4. Click "Cancel" to stop transcription

### Summarization

1. Click "Choose text" and select the transcript file
2. Click "Summarize" to generate a summary
3. The summary will be saved to `summary.txt` in the application directory
4. Click "Cancel" to stop summarization

## Configuration

Edit `config.json` to customize:
- Download URLs for binaries
- Model versions
- Model file locations

## Project Structure

```
llamawhisper/
├── app.py              # Main application
├── config.json         # Configuration file
├── system-summary.txt  # System prompt for summarization
├── models/             # Downloaded AI models
│   ├── ggml-medium.bin
│   └── Qwen3.5-9B-Q4_K_M.gguf
├── whisper/            # whisper.cpp binaries
└── llamacpp/           # llama.cpp binaries
```

## Technologies

- **whisper.cpp**: Efficient speech recognition in C/C++
- **llama.cpp**: LLM inference in pure C/C++
- **Qwen3.5**: Large language model for text summarization
- **Tkinter**: Python GUI toolkit

## License

This project uses open-source components:
- whisper.cpp (MIT License)
- llama.cpp (MIT License)
- Qwen3.5 (Apache 2.0 License)

---

# LlamaWhisper

Десктопное приложение для извлечения разговоров из звуковых файлов и форматирования текста в упорядоченные отчеты с использованием локальных ИИ-моделей. Создано с помощью языковой модели Qwen3.5-122B-A10B (Q6_K_L).

## Возможности

- **Расшифровка аудио**: Преобразование речи в текст с помощью whisper.cpp с поддержкой русского языка
- **Резюмирование текста**: Генерация кратких саммари транскрипций с помощью LLM Qwen3.5
- **Локальная обработка**: Все ИИ-модели работают локально — без облачных зависимостей
- **Отслеживание прогресса**: Индикаторы загрузки и обработки в реальном времени
- **Возможность отмены**: Прервите расшифровку или резюмирование в любой момент

## Требования

- Windows 10/11
- GPU NVIDIA с поддержкой CUDA (для ускорения инференса)
- Python 3.8+
- ~15 ГБ свободного места на диске для моделей и бинарников

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

## Использование

### Расшифровка

1. Нажмите "Choose audio" и выберите аудиофайл (поддерживаются `.wav`, `.mp3`, `.flac`, `.m4a`)
2. Нажмите "Transcribe" для начала расшифровки
3. Транскрипт будет сохранён как `.txt` файл рядом с аудиофайлом
4. Нажмите "Cancel" для остановки расшифровки

### Резюмирование

1. Нажмите "Choose text" и выберите файл транскрипта
2. Нажмите "Summarize" для генерации саммари
3. Саммари будет сохранено в `summary.txt` в директории приложения
4. Нажмите "Cancel" для остановки резюмирования

## Конфигурация

Отредактируйте `config.json` для настройки:
- URL-адресов загрузки бинарников
- Версий моделей
- Расположения файлов моделей

## Структура проекта

```
llamawhisper/
├── app.py              # Основное приложение
├── config.json         # Файл конфигурации
├── system-summary.txt  # Системный промпт для резюмирования
├── models/             # Скачанные ИИ-модели
│   ├── ggml-medium.bin
│   └── Qwen3.5-9B-Q4_K_M.gguf
├── whisper/            # Бинарники whisper.cpp
└── llamacpp/           # Бинарники llama.cpp
```

## Технологии

- **whisper.cpp**: Эффективное распознавание речи на C/C++
- **llama.cpp**: Инференс LLM на чистом C/C++
- **Qwen3.5**: Большая языковая модель для резюмирования текста
- **Tkinter**: GUI-тулкит для Python

## Лицензия

Проект использует компоненты с открытым исходным кодом:
- whisper.cpp (MIT License)
- llama.cpp (MIT License)
- Qwen3.5 (Apache 2.0 License)

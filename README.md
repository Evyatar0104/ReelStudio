# ReelStudio

Local video editing toolkit for content creators making high-volume short-form video (reels, shorts, TikToks). Built with Flask + faster-whisper + ffmpeg.

## Features

### Silence Cutter
- Detects and removes silent segments from video using ffmpeg `silencedetect`
- Parallel segment encoding (NVENC GPU or libx264 multi-threaded)
- Adjustable threshold, min duration, and padding
- Server-side file caching — tweak parameters without re-uploading

### Subtitle Transcriber
- Transcribes audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend)
- Context-aware subtitle splitting using word-level timestamps
- Configurable words-per-line and lines-per-subtitle
- Live subtitle preview overlaid on the video player
- Inline editor: Enter to split, merge, delete, reorder
- Hebrew-optimized with clause-aware line breaking
- SRT export with line wrapping

### Video Transcriber
- Full-text transcription with speaker diarization
- Markdown and SRT export

## Requirements

- **Python 3.10+**
- **ffmpeg** in PATH ([download](https://ffmpeg.org/download.html))
- **NVIDIA GPU** recommended (CUDA 12+) — falls back to CPU

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/ReelStudio.git
cd ReelStudio

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration, install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
python app.py
```

Open [http://localhost:5177](http://localhost:5177) in your browser.

**Windows quick start:** Double-click `start.bat` — it checks dependencies, installs what's missing, and launches the server + browser.

## Project Structure

```
ReelStudio/
├── app.py              # Flask backend — all API endpoints and workers
├── templates/
│   └── index.html      # Single-page frontend (inline CSS/JS)
├── requirements.txt    # Python dependencies
├── start.bat           # Windows launcher
├── uploads/            # Temp upload storage (gitignored)
├── outputs/            # Processed files (gitignored)
└── settings.json       # User preferences (gitignored)
```

## Performance Notes

- **NVENC auto-detection**: Uses GPU encoding if available, smart CPU threading otherwise
- **Parallel segment encoding**: N segments encoded concurrently (3 for NVENC, cores/2 for libx264)
- **Whisper VAD filter**: Silero VAD skips silent audio before the transformer
- **Greedy decoding**: `beam_size=1` for ~3x speed with minimal accuracy loss
- **File caching**: Upload once, reprocess with different parameters instantly
- **Silence detection cache**: Changing only padding skips the detection pass entirely

## License

MIT

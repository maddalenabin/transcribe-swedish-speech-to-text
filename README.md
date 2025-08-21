# Swedish Speech-to-Text Transcriber 🇸🇪

A Python application for transcribing Swedish audio files to text using the KBLab/kb-whisper-large model. Perfect for language learners who want to check their Swedish pronunciation!

## Motivation

As an immigrant learning Swedish, I created this tool to help evaluate my pronunciation. The idea is simple: if the AI can correctly transcribe my Swedish speech into proper text, it means my pronunciation is clear and understandable. This provides immediate feedback on how well I'm speaking Swedish without needing a native speaker present.

## Features

- **Two interfaces**: Command-line tool and web application
- **Swedish-optimized**: Uses KBLab's specialized Swedish Whisper model
- **Multiple audio formats**: Supports WAV, MP3, M4A, FLAC, OGG, AAC
- **Batch processing**: Process single files or entire directories
- **Fast web interface**: Model loads once, transcribe multiple files instantly
- **Download results**: Save transcriptions as text files

## Model Source
This project uses the **KBLab/kb-whisper-large** model from Hugging Face, developed by The National Library of Sweden. This model is specifically fine-tuned for Swedish speech recognition and provides significantly better accuracy for Swedish audio compared to the standard Whisper models.

- **Model**: [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large)
- **Organization**: KBLab, The National Library of Sweden
- **Base Model**: OpenAI Whisper Large
- **Specialization**: Swedish speech recognition

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/maddalenabin/transcribe-swedish-speech-to-text.git
cd transcribe-swedish-speech-to-text
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- torch
- transformers
- librosa
- soundfile
- flask (for web interface)

## Usage

### Option 1: Command-Line Interface

**Transcribe a single file:**
```bash
python swedish_transcriber.py audio.wav
```

**Save transcription to file:**
```bash
python swedish_transcriber.py audio.wav -o transcription.txt
```

**Process entire directory:**
```bash
python swedish_transcriber.py /path/to/audio/folder -o /path/to/output/folder
```

**Use custom model:**
```bash
python swedish_transcriber.py audio.wav --model KBLab/kb-whisper-large
```

### Option 2: Web Application (Recommended)

The web interface loads the model once and keeps it in memory, making subsequent transcriptions much faster.

#### Version 1: Fixed Port (swedish_transcriber_webapp.py)

1. **Start the web server:**
```bash
python swedish_transcriber_webapp.py
```

2. **Open your browser** and go to `http://localhost:5050`

#### Version 2: Automatic Port Detection (swedish_transcriber_webapp_v2.py)

1. **Start the web server:**
```bash
python swedish_transcriber_webapp_v2.py
```

2. **Open your browser** and go to the displayed URL (usually `http://localhost:5001`)

**Custom port (v2 only):**
```bash
python swedish_transcriber_webapp_v2.py --port 8080
```

**Make accessible from other devices (v2 only):**
```bash
python swedish_transcriber_webapp_v2.py --host 0.0.0.0 --port 8080
```

**Which version to use:**
- Use **v2** if you're having port conflicts or want flexible port options
- Use **v1** if you prefer a simple, fixed setup

## Examples

### Basic Transcription
```bash
# Transcribe a Swedish audio file
python swedish_transcriber.py examples/reading_rivstart.m4a

# Output: "Jag heter Anna och kommer från Sverige..."
```

### Pronunciation Check Workflow
1. **Record yourself** speaking Swedish (using your phone, computer, etc.)
2. **Run transcription**: Upload to web app or use command-line
3. **Check results**: If the transcription matches what you intended to say, your pronunciation is good!
4. **Identify issues**: Unclear transcriptions indicate pronunciation areas to work on

### Example for Language Learning
```bash
# You say: "Hej, jag heter Johan och jag kommer från Tyskland"
# Good transcription: "Hej, jag heter Johan och jag kommer från Tyskland"
# ✅ Your pronunciation is clear!

# You say: "Jag skulle vilja ha en kaffe"
# Transcription: "Jag skulle vilja ha en kaffe"  
# ✅ Perfect! The AI understood "kaffe" correctly

# You say: "Jag är trött" (but pronounce "trött" incorrectly)
# Transcription: "Jag är träd" or unclear text
# ❌ Need to work on the "ö" sound in "trött"
```

## File Structure

```
transcribe-swedish-speech-to-text/
├── README.md
├── requirements.txt
├── swedish_transcriber.py            # Command-line version
├── swedish_transcriber_webapp.py     # Web application (fixed port 5050)
├── swedish_transcriber_webapp_v2.py  # Web application (automatic port detection)
└── examples/
    ├── introduction.m4a              # Sample audio file
    └── reading_rivstart.m4a          # Sample reading audio
```

## Performance Notes

- **First run**: Model download (~3GB) and loading takes 3-5 minutes
- **Subsequent runs (CLI)**: Model loads each time (~2-3 minutes)
- **Web app**: Model loads once when starting server, then transcriptions are nearly instant
- **GPU acceleration**: Automatically uses CUDA if available for faster processing
- **Audio preprocessing**: Files are automatically resampled to 16kHz as required

## Troubleshooting

**Port already in use (macOS):**
- **For v1 (fixed port 5050)**: Disable AirPlay Receiver in System Preferences → General → AirDrop & Handoff, or use v2 instead
- **For v2 (automatic port)**: The app will automatically find an available port, or use `--port 8080` for a specific port

**Model loading errors:**
- Ensure stable internet connection for initial model download
- Check available disk space (model requires ~3GB)

**Audio file errors:**
- Try converting to WAV format if other formats fail
- Ensure audio file isn't corrupted

## Contributing

Feel free to open issues or submit pull requests to improve the tool!

## License

This project is open source. The KBLab/kb-whisper-large model follows its own licensing terms from Hugging Face.

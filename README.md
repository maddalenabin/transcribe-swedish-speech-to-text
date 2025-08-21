# First version: run from the terminal
### Installation Requirements:
First, install the required dependencies:
```bash
pip install torch transformers librosa soundfile
```

### Usage Examples:
Transcribe a single audio file:
```bash
python swedish_transcriber.py audio.wav
```

### Transcribe and save to text file:
```bash
python swedish_transcriber.py audio.wav -o transcription.txt
```

### Transcribe all audio files in a directory:
```bash
python swedish_transcriber.py /path/to/audio/folder -o /path/to/output/folder
```

# Second version: webpage to drag and drop all audio files to transcribe
### Installation Requirements:
First, install the required dependencies:
```bash
pip install torch transformers librosa soundfile
```

## Usage Examples:
### Option 1 - Automatic port (recommended):
Type:
```bash
python swedish_transcriber_webapp.py 
```
The app will automatically find a free port (usually 5001) and tell you the URL.

### Option 2 - Specify a custom port:
```bash
python swedish_transcriber_webapp.py --port 8080
```

### Additional option:
```bash
# Run on a specific port
python swedish_transcriber_webapp.py --port 3000

# Make accessible from other devices on your network
python swedish_transcriber_webapp.py --host 0.0.0.0 --port 8080
```
The app will now automatically find an available port and display the correct URL to open in your browser. This should resolve the "Address already in use" error you encountered!

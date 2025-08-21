#!/usr/bin/env python3
"""
Swedish Speech Transcription Web App
Flask web application with pre-loaded model for fast transcriptions
"""

import os
import tempfile
import time
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import warnings
from flask import Flask, request, render_template_string, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('transformers').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables for the model
transcriber = None
model_loading = True
model_load_error = None

class SwedishTranscriber:
    def __init__(self, model_name="KBLab/kb-whisper-large"):
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        print("Model loaded successfully!")
    
    def transcribe_audio_file(self, audio_path, language="sv"):
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features,
                    language=language,
                    task="transcribe"
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")

def load_model():
    """Load the model in a separate thread"""
    global transcriber, model_loading, model_load_error
    
    try:
        transcriber = SwedishTranscriber()
        model_loading = False
        print("Model ready for transcriptions!")
    except Exception as e:
        model_load_error = str(e)
        model_loading = False
        print(f"Error loading model: {e}")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Swedish Speech Transcriber</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .status {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }
        .status.loading {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .status.ready {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #3498db;
        }
        .upload-area.dragover {
            border-color: #3498db;
            background-color: #f8f9fa;
        }
        input[type="file"] {
            display: none;
        }
        .upload-btn {
            background-color: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .upload-btn:hover {
            background-color: #2980b9;
        }
        .upload-btn:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        .transcribe-btn {
            background-color: #27ae60;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 15px;
            transition: background-color 0.3s;
        }
        .transcribe-btn:hover {
            background-color: #229954;
        }
        .transcribe-btn:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        .result-area {
            margin-top: 30px;
        }
        .transcription {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 20px;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: Georgia, serif;
            line-height: 1.6;
            font-size: 16px;
        }
        .download-btn {
            background-color: #8e44ad;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
            text-decoration: none;
            display: inline-block;
        }
        .download-btn:hover {
            background-color: #7d3c98;
        }
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .file-info {
            margin-top: 15px;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 4px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇸🇪 Swedish Speech Transcriber</h1>
        
        <div id="modelStatus" class="status loading">
            Loading KBLab/kb-whisper-large model...
        </div>
        
        <div class="upload-area" id="uploadArea">
            <p>Drop an audio file here or click to select</p>
            <button type="button" class="upload-btn" id="uploadBtn" onclick="document.getElementById('audioFile').click()" disabled>
                Choose Audio File
            </button>
            <input type="file" id="audioFile" accept="audio/*" onchange="handleFileSelect(this)">
        </div>
        
        <div class="file-info" id="fileInfo">
            <strong>Selected file:</strong> <span id="fileName"></span>
        </div>
        
        <button type="button" class="transcribe-btn" id="transcribeBtn" onclick="transcribeAudio()" disabled>
            Transcribe Audio
        </button>
        
        <div class="loading-spinner" id="loadingSpinner"></div>
        
        <div class="result-area" id="resultArea" style="display: none;">
            <h3>Transcription Result:</h3>
            <div class="transcription" id="transcriptionResult"></div>
            <a href="#" class="download-btn" id="downloadBtn" style="display: none;">Download as Text File</a>
        </div>
    </div>

    <script>
        let selectedFile = null;
        let currentTranscription = null;

        // Check model status
        function checkModelStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('modelStatus');
                    const uploadBtn = document.getElementById('uploadBtn');
                    
                    if (data.loading) {
                        statusDiv.className = 'status loading';
                        statusDiv.textContent = 'Loading model... Please wait.';
                        uploadBtn.disabled = true;
                    } else if (data.error) {
                        statusDiv.className = 'status error';
                        statusDiv.textContent = `Error: ${data.error}`;
                        uploadBtn.disabled = true;
                    } else {
                        statusDiv.className = 'status ready';
                        statusDiv.textContent = '✓ Model ready! You can now upload audio files.';
                        uploadBtn.disabled = false;
                    }
                });
        }

        // Handle file selection
        function handleFileSelect(input) {
            if (input.files && input.files[0]) {
                selectedFile = input.files[0];
                document.getElementById('fileName').textContent = selectedFile.name;
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('transcribeBtn').disabled = false;
            }
        }

        // Handle drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                const fileInput = document.getElementById('audioFile');
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(fileInput);
            }
        });

        // Transcribe audio
        function transcribeAudio() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            // Show loading state
            document.getElementById('transcribeBtn').disabled = true;
            document.getElementById('loadingSpinner').style.display = 'block';
            document.getElementById('resultArea').style.display = 'none';
            
            fetch('/transcribe', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loadingSpinner').style.display = 'none';
                document.getElementById('transcribeBtn').disabled = false;
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    currentTranscription = data.transcription;
                    document.getElementById('transcriptionResult').textContent = data.transcription;
                    document.getElementById('resultArea').style.display = 'block';
                    document.getElementById('downloadBtn').style.display = 'inline-block';
                    document.getElementById('downloadBtn').href = '/download?text=' + encodeURIComponent(data.transcription) + '&filename=' + encodeURIComponent(selectedFile.name);
                }
            })
            .catch(error => {
                document.getElementById('loadingSpinner').style.display = 'none';
                document.getElementById('transcribeBtn').disabled = false;
                alert('Network error: ' + error);
            });
        }

        // Check model status on page load and periodically
        checkModelStatus();
        const statusInterval = setInterval(() => {
            checkModelStatus();
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (!data.loading) {
                        clearInterval(statusInterval);
                    }
                });
        }, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    global model_loading, model_load_error, transcriber
    return jsonify({
        'loading': model_loading,
        'ready': transcriber is not None,
        'error': model_load_error
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global transcriber
    
    if transcriber is None:
        return jsonify({'error': 'Model not loaded yet'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Transcribe
        start_time = time.time()
        transcription = transcriber.transcribe_audio_file(temp_path)
        processing_time = time.time() - start_time
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return jsonify({
            'transcription': transcription,
            'processing_time': f"{processing_time:.2f} seconds"
        })
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/download')
def download():
    text = request.args.get('text', '')
    filename = request.args.get('filename', 'transcription')
    
    if not text:
        return "No text to download", 400
    
    # Create temporary text file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    temp_file.write(text)
    temp_file.close()
    
    # Generate download filename
    base_name = Path(filename).stem
    download_filename = f"{base_name}_transcription.txt"
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=download_filename,
        mimetype='text/plain'
    )

def main():
    print("Starting Swedish Transcriber Web App...")
    print("Model will be loaded in the background...")
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=load_model)
    model_thread.daemon = True
    model_thread.start()
    
    print("\n" + "="*50)
    print("🚀 Web app starting!")
    print("📝 Open your browser and go to: http://localhost:5050")
    print("⏳ The model will load in the background...")
    print("✅ Once loaded, you can transcribe audio files instantly!")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)

if __name__ == "__main__":
    main()
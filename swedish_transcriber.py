#!/usr/bin/env python3
"""
Swedish Speech Transcription Tool
Uses KBLab/kb-whisper-large model to transcribe Swedish audio files
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class SwedishTranscriber:
    def __init__(self, model_name="KBLab/kb-whisper-large"):
        """Initialize the Swedish transcriber with the specified model."""
        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run...")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the processor and model
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def load_audio(self, audio_path, sample_rate=16000):
        """Load and preprocess audio file."""
        try:
            print(f"Loading audio file: {audio_path}")
            # Load audio file and resample to 16kHz (Whisper's expected sample rate)
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            print(f"Audio loaded: {len(audio)/sample_rate:.2f} seconds")
            return audio
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
    
    def transcribe_audio(self, audio_path, language="sv"):
        """Transcribe audio file to Swedish text."""
        # Load audio
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        
        try:
            print("Processing audio...")
            
            # Process audio with the processor
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            print("Generating transcription...")
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features,
                    language=language,
                    task="transcribe"
                )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def transcribe_file(self, input_path, output_path=None):
        """Transcribe a single audio file and optionally save to file."""
        if not os.path.exists(input_path):
            print(f"Error: Audio file '{input_path}' not found.")
            return False
        
        # Transcribe the audio
        transcription = self.transcribe_audio(input_path)
        
        if transcription is None:
            print("Transcription failed.")
            return False
        
        print(f"\n--- Transcription ---")
        print(transcription)
        print(f"--- End Transcription ---\n")
        
        # Save to file if output path is specified
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Transcription saved to: {output_path}")
            except Exception as e:
                print(f"Error saving transcription: {e}")
                return False
        
        return True
    
    def transcribe_directory(self, input_dir, output_dir=None):
        """Transcribe all audio files in a directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Directory '{input_dir}' not found.")
            return
        
        # Common audio file extensions
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        audio_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"No audio files found in '{input_dir}'")
            return
        
        print(f"Found {len(audio_files)} audio files to transcribe.")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each audio file
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            # Determine output file path
            if output_dir:
                output_file = output_path / f"{audio_file.stem}_transcription.txt"
            else:
                output_file = None
            
            self.transcribe_file(str(audio_file), str(output_file) if output_file else None)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Swedish audio files using KBLab/kb-whisper-large model"
    )
    parser.add_argument(
        "input", 
        help="Input audio file or directory containing audio files"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (for single file) or directory (for multiple files)"
    )
    parser.add_argument(
        "--model",
        default="KBLab/kb-whisper-large",
        help="Model name to use (default: KBLab/kb-whisper-large)"
    )
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = SwedishTranscriber(args.model)
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Transcribe single file
        transcriber.transcribe_file(args.input, args.output)
    elif input_path.is_dir():
        # Transcribe all files in directory
        transcriber.transcribe_directory(args.input, args.output)
    else:
        print(f"Error: '{args.input}' is not a valid file or directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
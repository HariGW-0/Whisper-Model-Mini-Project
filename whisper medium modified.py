import os
import sys
import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from ttkthemes import ThemedTk
import threading
import queue
from datetime import datetime

# Configuration
MODEL_NAME = "/home/art/Pictures/whisper medium"  # Official model name for auto-download
SAMPLE_RATE = 16000
CHUNK_DURATION = 10  # Seconds
LANGUAGE = "en"

# Initialize model with proper error handling
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    
    if device == "cuda":
        model = model.half()
        
    print("Model loaded successfully")

except Exception as e:
    print(f"ERROR LOADING MODEL: {str(e)}")
    sys.exit("Check your internet connection and try again")

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.geometry("600x400")
        
        # Create tabs
        self.tab_control = ttk.Notebook(root)
        self.tab_control.pack(expand=1, fill="both")
        
        # Control Panel Tab
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text="Controls")
        
        # Transcription Output Tab
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2, text="Transcriptions")
        
        # Control Panel UI
        self.start_btn = ttk.Button(
            self.tab1,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.start_btn.pack(pady=20)
        
        self.status_label = ttk.Label(
            self.tab1,
            text="Status: Idle",
            font=("Helvetica", 12)
        )
        self.status_label.pack(pady=10)
        
        self.current_transcription = scrolledtext.ScrolledText(
            self.tab1,
            width=60,
            height=5,
            state='disabled',
            wrap=tk.WORD
        )
        self.current_transcription.pack(pady=10)
        
        # Transcription Output UI
        self.output_text = scrolledtext.ScrolledText(
            self.tab2,
            width=70,
            height=20,
            state='disabled',
            wrap=tk.WORD
        )
        self.output_text.pack(padx=10, pady=10)
        
        # Recording state
        self.recording = False
        self.thread = None

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.start_btn.config(text="Stop Recording")
        self.status_label.config(text="Status: Recording...", foreground="red")
        
        # Clear UI
        self.current_transcription.config(state='normal')
        self.current_transcription.delete(1.0, tk.END)
        self.current_transcription.config(state='disabled')
        
        # Start background thread
        self.thread = threading.Thread(target=self.record_and_transcribe)
        self.thread.daemon = True
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        self.start_btn.config(text="Start Recording")
        self.status_label.config(text="Status: Idle", foreground="black")

    def record_and_transcribe(self):
        while self.recording:
            # Record audio chunk
            audio = sd.rec(
                int(CHUNK_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            if not self.recording:
                break
            
            # Process audio
            audio = audio.flatten()
            audio = self.normalize_audio(audio)
            
            # Debug output
            print(f"Audio max: {np.abs(audio).max():.4f}")
            
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_features.to(device)
            
            if device == "cuda":
                inputs = inputs.half()
            
            # Generate transcription
            try:
                with torch.no_grad():
                    predicted_ids = model.generate(
                        inputs,
                        max_length=200,
                        num_beams=3,
                        temperature=0.5,
                        repetition_penalty=1.5,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(language=LANGUAGE),
                        do_sample=True,
                        length_penalty=0.5
                    )
                    
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                print(f"Transcribed: {text}")
                
                # Update UI
                self.root.after(0, self.update_ui, text)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                self.root.after(0, self.update_ui, "[Error]")

    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val == 0:
            return audio
        return audio / max_val

    def update_ui(self, text):
        # Update current transcription
        self.current_transcription.config(state='normal')
        self.current_transcription.delete(1.0, tk.END)
        self.current_transcription.insert(tk.END, text)
        self.current_transcription.config(state='disabled')
        
        # Update history log
        self.output_text.config(state='normal')
        self.output_text.insert(
            tk.END,
            f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n\n"
        )
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = TranscriptionApp(root)
    root.mainloop()

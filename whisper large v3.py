import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configuration
MODEL_PATH = "/home/art/Downloads/Whisper large v3"  # Update path to your model location
SAMPLE_RATE = 16000  # 16kHz sample rate
CHUNK_DURATION = 10  # Duration (in seconds) for each audio chunk
LANGUAGE = "en"

# Device selection and AMP flag
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"  # Enable automatic mixed precision on CUDA

# Load processor and model
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
if device == "cuda":
    model = model.half()  # Convert model to half precision for CUDA

def record_and_transcribe():
    """Record CHUNK_DURATION-second chunks and transcribe them."""
    try:
        print(f"=== Recording {CHUNK_DURATION}-second chunks (Press CTRL+C to stop) ===")
        while True:
            print("\nRecording...")
            try:
                audio = sd.rec(
                    int(CHUNK_DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()  # Wait until recording completes
            except Exception as e:
                print(f"Audio recording error: {e}")
                continue

            # Pre-process audio data
            audio = audio.flatten()
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_features.to(device)

            if device == "cuda":
                inputs = inputs.half()

            # Transcription generation with optional AMP context for improved performance
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predicted_ids = model.generate(
                            inputs,
                            max_length=200,
                            num_beams=3,
                            temperature=0.5,
                            repetition_penalty=1.5,
                            forced_decoder_ids=processor.get_decoder_prompt_ids(language=LANGUAGE)
                        )
                else:
                    predicted_ids = model.generate(
                        inputs,
                        max_length=200,
                        num_beams=3,
                        temperature=0.5,
                        repetition_penalty=1.5,
                        forced_decoder_ids=processor.get_decoder_prompt_ids(language=LANGUAGE)
                    )

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"\nTranscription: {text.strip()}")

    except KeyboardInterrupt:
        print("\n=== Transcription stopped ===")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

if __name__ == "__main__":
    record_and_transcribe()

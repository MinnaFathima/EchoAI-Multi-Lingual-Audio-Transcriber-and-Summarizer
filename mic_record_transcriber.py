import os
import tempfile
import whisper
import sounddevice as sd
import scipy.io.wavfile as wavfile
from transformers import pipeline
import keyboard
import numpy as np

LANGUAGE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'kn': 'Kannada', 'ml': 'Malayalam',
    'mr': 'Marathi', 'gu': 'Gujarati', 'bn': 'Bengali', 'pa': 'Punjabi', 'ur': 'Urdu', 'zh': 'Chinese',
    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ar': 'Arabic', 'ru': 'Russian'
}

# === Record from Microphone (start/stop by key) ===
def record_audio_dynamic(filename, samplerate=16000):
    print("\n🎙️ Press 's' to start recording and 'e' to stop...")

    while True:
        if keyboard.is_pressed('s'):
            print("🎬 Recording started... Press 'e' to stop.")
            recording = []
            while True:
                if keyboard.is_pressed('e'):
                    print("🛑 Recording stopped.")
                    break
                chunk = sd.rec(int(0.5 * samplerate), samplerate=samplerate, channels=1, dtype='int16')
                sd.wait()
                recording.append(chunk)
            audio = np.concatenate(recording, axis=0)
            wavfile.write(filename, samplerate, audio)
            print(f"✅ Audio saved to {filename}\n")
            return

if __name__ == "__main__":
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = temp_audio.name
    record_audio_dynamic(audio_path)

    # === Load Whisper Model ===
    print("🔁 Loading Whisper model...")
    model = whisper.load_model("base")

    # === Transcribe and Translate ===
    print("🗣️ Transcribing and translating...")
    result = model.transcribe(audio_path, task="translate")
    lang_code = result.get("language", "unknown")
    lang_name = LANGUAGE_MAP.get(lang_code, lang_code.title())
    print(f"\n🌐 Detected language: {lang_name}")

    full_text = result["text"]
    print("\n📝 Transcribed Text:\n")
    print(full_text)

    # === Summarize ===
    print("\n🧠 Summarizing...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
    summary = ""
    for chunk in chunks:
        out = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summary += out[0]['summary_text'] + "\n"

    print("\n✅ Final Summary:\n")
    print(summary)

    # === Save Summary ===
    summary_file = os.path.join(os.path.dirname(audio_path), "mic_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n📝 Summary saved to: {summary_file}")

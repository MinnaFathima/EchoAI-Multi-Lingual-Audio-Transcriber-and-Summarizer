import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog
import whisper
from transformers import pipeline

LANGUAGE_MAP = {
    'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'as': 'Assamese', 'az': 'Azerbaijani', 'ba': 'Bashkir',
    'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian',
    'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish',
    'fo': 'Faroese', 'fr': 'French', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian',
    'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'ht': 'Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian',
    'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'ja': 'Japanese', 'jw': 'Javanese', 'ka': 'Georgian',
    'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'la': 'Latin', 'lb': 'Luxembourgish',
    'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori',
    'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese',
    'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'nn': 'Nynorsk', 'no': 'Norwegian', 'oc': 'Occitan',
    'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
    'sa': 'Sanskrit', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sn': 'Shona',
    'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili',
    'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tk': 'Turkmen', 'tl': 'Tagalog',
    'tr': 'Turkish', 'tt': 'Tatar', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese',
    'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese'
}


# === Select file ===
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select Audio or Video File",
    filetypes=[("Media files", "*.mp3 *.wav *.m4a *.mp4 *.mkv *.avi *.mov")]
)

if not file_path:
    print("❌ No file selected.")
    exit()

# === Convert video to audio if needed ===
ext = os.path.splitext(file_path)[1].lower()
if ext in [".mp4", ".mkv", ".avi", ".mov"]:
    print("🎬 Extracting audio from video...")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_path = temp_audio.name

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", file_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-y",
        audio_path
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    audio_path = file_path

# === Load Whisper model ===
print("🔁 Loading Whisper model...")
model = whisper.load_model("base")

# === Transcribe with translation to English ===
print("🎙️ Transcribing and translating...")
result = model.transcribe(audio_path, task="translate")

# === Show detected language ===
lang = result.get("language", "unknown")
print(f"\n🌐 Detected language: {lang.title()}")

# === Print transcript ===
full_text = result["text"]
print("\n🗣️ Transcribed Text:\n")
print(full_text)

# === Summarize ===
print("\n🧠 Summarizing...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
summary = ""
for chunk in chunks:
    out = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
    summary += out[0]["summary_text"] + "\n"

print("\n✅ Final Summary:\n")
print(summary)

# === Save summary ===
summary_file = os.path.join(os.path.dirname(file_path), "summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\n📝 Summary saved to: {summary_file}")

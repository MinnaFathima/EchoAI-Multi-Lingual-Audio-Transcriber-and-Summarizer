import os
import tempfile
import whisper
import gradio as gr
import subprocess
from transformers import pipeline

LANGUAGE_MAP = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'kn': 'Kannada', 'ml': 'Malayalam',
    'mr': 'Marathi', 'gu': 'Gujarati', 'bn': 'Bengali', 'pa': 'Punjabi', 'ur': 'Urdu', 'zh': 'Chinese',
    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ar': 'Arabic', 'ru': 'Russian'
}

whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def handle_file_input(file_obj):
    file_path = file_obj.name
    ext = os.path.splitext(file_path)[1].lower()

    # Convert video to audio
    if ext in [".mp4", ".mkv", ".avi", ".mov"]:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_path = temp_audio.name
        cmd = ["ffmpeg", "-i", file_path, "-vn", "-acodec", "libmp3lame", "-y", audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        audio_path = file_path

    return transcribe_and_summarize(audio_path)

def handle_audio_input(audio_path):
    return transcribe_and_summarize(audio_path)

def transcribe_and_summarize(audio_path):
    result = whisper_model.transcribe(audio_path, task="translate")
    lang_code = result.get("language", "unknown")
    lang = LANGUAGE_MAP.get(lang_code, lang_code.upper())
    transcript = result["text"]

    chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
    summary = ""
    for chunk in chunks:
        out = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summary += out[0]["summary_text"] + "\n"

    summary_path = os.path.join(os.path.dirname(audio_path), "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return lang, transcript, summary, summary_path

with gr.Blocks() as demo:
    gr.Markdown("# 🎧 Whisper Transcriber & Summarizer")

    with gr.Tabs():
        with gr.TabItem("🎤 Record from Mic"):
            mic_input = gr.Audio(type="filepath", label="🎤 Record or Upload Audio")
            mic_button = gr.Button("Transcribe & Summarize")
            mic_lang = gr.Textbox(label="🌐 Detected Language")
            mic_text = gr.Textbox(label="📝 Transcript")
            mic_summary = gr.Textbox(label="🧠 Summary")
            mic_file = gr.File(label="📥 Download Summary")

            mic_button.click(handle_audio_input, inputs=mic_input, outputs=[mic_lang, mic_text, mic_summary, mic_file])

        with gr.TabItem("📁 Upload Audio/Video File"):
            file_input = gr.File(label="Upload Audio/Video", file_types=[".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".avi", ".mov"])
            file_button = gr.Button("Transcribe & Summarize")
            file_lang = gr.Textbox(label="🌐 Detected Language")
            file_text = gr.Textbox(label="📝 Transcript")
            file_summary = gr.Textbox(label="🧠 Summary")
            file_file = gr.File(label="📥 Download Summary")

            file_button.click(handle_file_input, inputs=file_input, outputs=[file_lang, file_text, file_summary, file_file])

demo.launch()

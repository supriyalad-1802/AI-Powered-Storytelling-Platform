import os
import pathlib
import tempfile
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from dotenv import load_dotenv
from fpdf import FPDF
from werkzeug.utils import secure_filename

# Load environment variables from .env file FIRST before doing anything else
load_dotenv()

from speech import text_to_audio_file
from event_detection import check_file_domain, get_ner_entities_simple
from captioning import caption_video, caption_image_file, format_video_captions, is_cricket_visual
from summarizer import build_fused_document, assess_data_sufficiency, generate_story, build_template_narrative
from video_utils import extract_highlight

# ── Flask setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "frontend", "static"),
)
app.secret_key = "cricket_ai_secret_2026"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

RESULTS_ROOT = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)

ALLOWED_EXT = {
    "videos": {"mp4", "mov", "avi"},
    "audios": {"mp3", "wav"},
    "images": {"jpeg", "jpg", "png"},
    "texts":  {"txt", "md"},
}

def get_ftype(filename: str):
    """Determine the file type category based on its extension."""
    ext = pathlib.Path(filename).suffix.lower().lstrip(".")
    for ft, exts in ALLOWED_EXT.items():
        if ext in exts:
            return ft
    return None

def wfile(path: str, text: str):
    """Helper to write text to a file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)

# ── Model loading ──────────────────────────────────────────────────────────────
print("\n🔹 Loading AI models...\n")

try:
    import whisper as _whisper
    whisper_model = _whisper.load_model("tiny")
    print("✅ Whisper loaded (tiny).")
except Exception as e:
    whisper_model = None
    print(f"❌ Whisper failed: {e}")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.to("cpu").eval()
    print("✅ BLIP loaded.")
except Exception as e:
    blip_processor = blip_model = None
    print(f"❌ BLIP failed to load: {e}")

try:
    import easyocr
    ocr_reader = easyocr.Reader(['en'])
    print("✅ EasyOCR loaded.")
except Exception as e:
    ocr_reader = None
    print(f"❌ EasyOCR failed to load: {e}")

print("\nPlatform ready.\n")

# ── Transcription helper ───────────────────────────────────────────────────────
def transcribe_file(path: str):
    """Transcribe an audio or video file using Whisper, returning plain text and timestamped text."""
    if not whisper_model: return "", "", None
    print(f"  ▶ Transcribing: {os.path.basename(path)}")
    try:
        result = whisper_model.transcribe(path, fp16=False)
        plain = result.get("text", "").strip()
        lang  = result.get("language", None)
        stamped = "\n".join(f"[{int(s['start'])//60:02d}:{int(s['start'])%60:02d}] {s['text'].strip()}" for s in result.get("segments", []))
        return plain, stamped, lang
    except Exception as e:
        return "", "", None

# ── Results saver ──────────────────────────────────────────────────────────────
def save_results(event: str, data: dict) -> str:
    """Save processed data and generated narrative to the results directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_event = "".join(c if c.isalnum() or c in "-_" else "_" for c in event)
    folder = os.path.join(RESULTS_ROOT, f"{safe_event}_{ts}")
    os.makedirs(folder, exist_ok=True)
    fl = ", ".join(data.get("files_processed", []))

    if data.get("transcript"): wfile(os.path.join(folder, "transcript.txt"), f"Files: {fl}\n\n{data['transcript']}")
    if data.get("ts_transcript"): wfile(os.path.join(folder, "transcript_timestamped.txt"), data["ts_transcript"])
    if data.get("video_caps"): wfile(os.path.join(folder, "video_captions.txt"), data["video_caps"])
    if data.get("image_caps"): wfile(os.path.join(folder, "image_captions.txt"), data["image_caps"])
    if data.get("text_inputs"): wfile(os.path.join(folder, "text_inputs.txt"), data["text_inputs"])
    if data.get("narrative"): wfile(os.path.join(folder, "cricket_story_narrative.txt"), f"=== CRICKET MATCH STORY ===\nEvent: {event}\nGenerated: {ts}\n\n{data['narrative']}")
    return os.path.basename(folder)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    """Render the main upload page."""
    return render_template("upload.html", narrative=None, not_enough_msg=None, domain_ok=None, autoplay=False, highlight_video=None)

@app.route("/upload", methods=["POST"])
def upload():
    """Handle the media upload and orchestration pipeline."""
    try:
        event = request.form.get("event_name", "").strip()
        duration = int(request.form.get("duration", 2))
        custom_prompt = request.form.get("custom_prompt", "").strip()
        language = request.form.get("language", "English")
        autoplay = request.form.get("autoplay") == "on" # Capture user's auto-play choice
        
        if not event:
            flash("Event name is required.", "error")
            return redirect(url_for("home"))

        files = request.files.getlist("media")
        if not files or all(f.filename == "" for f in files):
            flash("No files selected.", "error")
            return redirect(url_for("home"))

        with tempfile.TemporaryDirectory() as tmp:
            plains, stampeds, v_caps, i_caps, t_contents = [], [], [], [], []
            files_done, rejected, videos_with_speech = [], [], set()
            video_filepath = None

            saved = []
            for f in files:
                if not f.filename: continue
                ft = get_ftype(f.filename)
                if not ft: continue
                safe = secure_filename(f.filename)
                dest = os.path.join(tmp, safe)
                f.save(dest)
                saved.append((ft, safe, dest))

            if not saved:
                flash("No valid files uploaded.", "error")
                return redirect(url_for("home"))

            for ft, name, path in saved:
                if ft in ("videos", "audios"):
                    if ft == "videos" and not video_filepath:
                        video_filepath = path
                    plain, stamped, detected_lang = transcribe_file(path)
                    if not plain:
                        if ft == "videos" and blip_model:
                            caps = caption_video(path, blip_processor, blip_model)
                            if caps:
                                v_caps.extend(caps)
                                files_done.append(name)
                            else: rejected.append(name)
                        continue
                    dc = check_file_domain(plain, name, is_transcript=True)
                    if dc["reject"]:
                        rejected.append(name)
                        continue
                    plains.append(f"[{name}]\n{plain}")
                    stampeds.append(f"[{name}]\n{stamped}")
                    files_done.append(name)
                    if ft == "videos": videos_with_speech.add(name)

                elif ft == "texts":
                    try: content = open(path, encoding="utf-8", errors="replace").read().strip()
                    except Exception: continue
                    if not content: continue
                    dc = check_file_domain(content, name, is_transcript=False)
                    if dc["reject"]:
                        rejected.append(name)
                        continue
                    t_contents.append((name, content))
                    files_done.append(name)

            for ft, name, path in saved:
                if name in rejected: continue
                if ft == "videos" and blip_model:
                    if name not in videos_with_speech and name not in files_done:
                        caps = caption_video(path, blip_processor, blip_model)
                        if caps:
                            v_caps.extend(caps)
                            files_done.append(name)
                        else: rejected.append(name)
                    elif name in videos_with_speech:
                        caps = caption_video(path, blip_processor, blip_model)
                        if caps: v_caps.extend(caps)
                elif ft == "images":
                    # 1. OCR Pass (read scorecards)
                    if ocr_reader:
                        try:
                            results = ocr_reader.readtext(path, detail=0)
                            if results:
                                text_found = " ".join(results)
                                t_contents.append((f"{name} (OCR Scorecard)", text_found))
                                print(f"  [OCR] Extracted {len(text_found)} chars from {name}")
                        except Exception as e:
                            print(f"  [OCR Error] {e}")

                    # 2. BLIP Pass (read visual action)
                    if not blip_model: continue
                    raw_cap, context_cap = caption_image_file(path, blip_processor, blip_model, len(i_caps))
                    if not raw_cap or not context_cap: continue
                    if is_cricket_visual(raw_cap):
                        i_caps.append((name, context_cap))
                        if name not in files_done: files_done.append(name)
                    else: 
                        if name not in rejected: rejected.append(name)

            has_content = bool(plains or t_contents or v_caps or i_caps)
            if not has_content:
                flash("No valid cricket content extracted.", "error")
                return redirect(url_for("home"))

            combined_transcript = "\n\n".join(plains)
            suf = assess_data_sufficiency(combined_transcript, t_contents, v_caps, i_caps)
            if not suf["sufficient"]:
                flash("Not enough data to generate a story.", "warning")
                return redirect(url_for("home"))

            ner_src = combined_transcript + " " + " ".join(c for _, c in t_contents)
            entities = get_ner_entities_simple(ner_src)
            fused = build_fused_document(combined_transcript, v_caps, i_caps, t_contents, entities)
            
            narrative, err = generate_story(
                fused_doc=fused, transcript=combined_transcript, video_captions=v_caps,
                image_captions=i_caps, text_contents=t_contents, entities=entities,
                event_name=event, target_minutes=duration, language=language, custom_instruction=custom_prompt
            )

            if err or not narrative:
                narrative = build_template_narrative(combined_transcript, v_caps, i_caps, t_contents, entities, event)

            highlight_video = None
            if video_filepath and "[HIGHLIGHT_TIMESTAMP]" in narrative:
                # Extract timestamp and splice video
                match = re.search(r'\[HIGHLIGHT_TIMESTAMP\]:\s*(\d{2}:\d{2})', narrative)
                if match:
                    timestamp = match.group(1)
                    highlight_video_name = extract_highlight(video_filepath, timestamp, app.static_folder)
                    if highlight_video_name:
                        highlight_video = f"results/{highlight_video_name}"

            folder = save_results(event, {
                "transcript": combined_transcript, "ts_transcript": "\n\n".join(stampeds),
                "video_caps": format_video_captions(v_caps), "image_caps": "\n".join(f"[{n}] {c}" for n, c in i_caps),
                "text_inputs": "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c in t_contents),
                "narrative": narrative, "files_processed": files_done,
            })

        return render_template(
            "upload.html",
            narrative=narrative,
            not_enough_msg=None,
            domain_ok=True,
            result_folder=folder,
            entities=entities,
            autoplay=autoplay,
            language=language,
            highlight_video=highlight_video
        )
    except Exception as e:
        print(f"Server Error in /upload: {str(e)}")
        flash("An unexpected error occurred processing your request.", "error")
        return redirect(url_for("home"))

@app.route("/listen", methods=["POST"])
def listen():
    """Synthesize text to audio using edge-tts."""
    data = request.get_json()
    narrative = data.get("narrative")
    language = data.get("language", "English")
    if not narrative: return jsonify({"error": "No narrative provided"}), 400
        
    # 1. Remove markdown symbols
    clean_narrative = re.sub(r'[*_#]', '', narrative)
    
    # 2. Remove the Highlight Timestamp and anything that comes after it
    clean_narrative = re.sub(r'\[HIGHLIGHT_TIMESTAMP\][\s\S]*', '', clean_narrative)
    
    # 3. Handle SSML tags so they are not read aloud
    # Replace <break> tags with ellipses for a natural pause
    clean_narrative = re.sub(r'<break[^>]*>', '... ', clean_narrative)
    # Strip <prosody> and any other XML tags that the TTS engine might try to read literally
    clean_narrative = re.sub(r'<[^>]*>', '', clean_narrative)
    clean_narrative = clean_narrative.strip()
    
    audio_path = os.path.join(tempfile.gettempdir(), "cricket_story.mp3")
    output_file, err = text_to_audio_file(clean_narrative, audio_path, language)
    
    if err:
        print(f"\n❌ TTS ERROR: {err}\n")
        return jsonify({"error": err}), 500
        
    return send_file(output_file, mimetype="audio/mpeg")

@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    data = request.get_json()
    narrative = data.get("narrative", "")
    event_name = data.get("event_name", "Match_Report")
    if not narrative: return jsonify({"error": "No text provided"}), 400

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"AI Cricket Storyteller: {event_name}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    safe_text = narrative.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    
    pdf_path = os.path.join(tempfile.gettempdir(), f"{event_name}.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
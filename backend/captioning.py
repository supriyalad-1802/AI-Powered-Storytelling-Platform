

import cv2
import numpy as np
from PIL import Image
import re


# ─────────────────────────────────────────────────────────────────────────────
# Caption cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_caption(text: str):
   
    if not text:
        return None
    text = text.strip()

    # Reject social-media noise
    bad = ["twitter", "instagram", "@", "image tagged", "http", "www.", "flickr"]
    if any(b in text.lower() for b in bad):
        return None

    # Strip leading punctuation
    text = re.sub(r'^[,.\s]+', '', text).strip()

    # Deduplicate comma-separated repeated phrases
    parts = [p.strip() for p in text.split(",")]
    seen = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    text = ", ".join(seen)

    # Deduplicate consecutive repeated words
    words = text.split()
    if len(words) > 3:
        deduped = [words[0]]
        for i in range(1, len(words)):
            if words[i].lower() != words[i - 1].lower():
                deduped.append(words[i])
        text = " ".join(deduped)

    # FIX: lowered minimum from 6 → 4 chars so short valid captions survive
    if len(text.strip()) < 4:
        return None

    # Capitalise and ensure terminal punctuation
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Cricket context map
# ─────────────────────────────────────────────────────────────────────────────

CRICKET_CONTEXT = {
    "crowd":     "Large crowd cheering in the cricket stadium",
    "audience":  "Spectators gathered at the cricket ground",
    "trophy":    "Players celebrating with the championship trophy",
    "cup":       "Team lifting the winners cup in celebration",
    "hug":       "Players embracing in celebration on the field",
    "embrace":   "Teammates celebrating together on the cricket ground",
    "player":    "Cricket player on the field",
    "players":   "Cricket players on the field",
    "men":       "Cricketers on the cricket ground",
    "man":       "Cricketer on the field",
    "ball":      "Cricket ball in play",
    "bat":       "Batsman at the crease",
    "stadium":   "Cricket stadium during the match",
    "field":     "Cricket ground during the match",
    "pitch":     "Cricket pitch view",
    "wicket":    "Wicket area at the crease",
    "run":       "Batsman running between the wickets",
    "catch":     "Fielder attempting a catch",
    "cheer":     "Crowd cheering a cricket moment",
    "celebrate": "Team celebrating a cricket milestone",
    "lift":      "Players lifting the trophy in celebration",
    "flag":      "Supporters waving flags in the stands",
    "jersey":    "Cricketers in their match jerseys",
    "helmet":    "Batsman wearing protective gear at the crease",
    "grass":     "Players on the cricket outfield",
    "green":     "Cricket ground visible",
    "white":     "Players in white cricket clothing",
}

# Keywords that strongly suggest the image is NOT cricket
NON_CRICKET_VISUAL_KEYWORDS = [
    "football", "soccer", "basketball", "tennis", "golf", "swimming",
    "boxing", "rugby", "volleyball", "badminton", "hockey puck",
    "formula", "race car", "racing", "surfing", "skiing",
    "cat", "dog", "food", "pizza", "burger", "cake", "car",
    "kitchen", "bedroom", "office", "classroom", "hospital",
    "nature", "mountain", "beach", "ocean", "forest",
]


def is_cricket_visual(raw_caption: str) -> bool:
    if not raw_caption:
        return False

    lower = raw_caption.lower()

    # Still reject obvious other sports or completely unrelated things
    for kw in NON_CRICKET_VISUAL_KEYWORDS:
        if kw in lower:
            print(f"    [Domain] Image rejected — non-cricket visual: '{kw}' in caption")
            return False

    # RELAXED RULE: If it didn't trigger a strict non-cricket word, accept it!
    # The LLM will weave it into the story if it makes sense.
    print(f"    [Domain] Image accepted into pipeline: '{raw_caption}'")
    return True
    # Cricket keywords present — accept
    CRICKET_VISUAL_WORDS = [
        "cricket", "bat", "ball", "wicket", "stadium", "pitch", "crease",
        "crowd", "player", "players", "field", "grass", "jersey",
        "helmet", "trophy", "cup", "celebrate", "cheer", "flag",
        "man", "men", "white", "sport", "game", "match",
        "green", "run", "catch", "umpire",
    ]
    for kw in CRICKET_VISUAL_WORDS:
        if kw in lower:
            return True

    # Ambiguous — accept with a note (better to include than silently skip)
    print(f"    [Domain] Image ambiguous, included: '{raw_caption}'")
    return True


def add_cricket_context(caption: str) -> str:
   
    if not caption:
        return caption
    lower = caption.lower()
    for kw, ctx in CRICKET_CONTEXT.items():
        if kw in lower:
            return ctx
    return f"Cricket scene: {caption}"


# ─────────────────────────────────────────────────────────────────────────────
# Video frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, every_n_sec: int = 5, max_frames: int = 4):
    """Extract representative frames from a video, skipping similar frames."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Frame extract] Cannot open video: {video_path}")
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"  [Video] {duration:.1f}s, {fps:.1f}fps, {total} frames")

    interval = (
        3 if duration <= 30
        else (every_n_sec if duration <= 120
              else max(5, int(duration / max_frames)))
    )
    fi = max(1, int(fps * interval))
    prev_gray, idx = None, 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % fi == 0:
            ts = idx / fps
            try:
                gray = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)
                )
                # Skip visually identical frames
                if (prev_gray is not None and
                        np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))) < 8.0):
                    idx += 1
                    continue
                prev_gray = gray.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((ts, Image.fromarray(rgb)))
            except Exception as e:
                print(f"  [Frame extract] Error at frame {idx}: {e}")
        idx += 1

    cap.release()
    print(f"  [Frame extract] {len(frames)} frames extracted")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Core BLIP inference
# ─────────────────────────────────────────────────────────────────────────────

def caption_one(blip_processor, blip_model, image: Image.Image):
   
    try:
        # Ensure RGB — BLIP fails on RGBA or grayscale
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model's expected input size
        img = image.resize((384, 384), Image.LANCZOS)

        # FIX: Use images= keyword (matches BlipProcessor API)
        inputs = blip_processor(images=img, return_tensors="pt")

        # Move to same device as model
        device = next(blip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = blip_model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=1,
            repetition_penalty=3.0,
            no_repeat_ngram_size=3,
            min_length=5,
        )

        raw = blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"    [BLIP raw] {repr(raw)}")

        cleaned = clean_caption(raw)
        if not cleaned:
            print(f"    [BLIP] Caption cleaned to None — raw was: {repr(raw)}")
        return cleaned

    except Exception as e:
        print(f"    [BLIP ERROR] {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Video captioning
# ─────────────────────────────────────────────────────────────────────────────

def caption_video(video_path: str, blip_processor, blip_model):
   
    if blip_model is None or blip_processor is None:
        print("  [BLIP] Model not loaded — video captioning skipped")
        return []

    frames = extract_frames(video_path)
    if not frames:
        print("  [BLIP] No frames extracted from video")
        return []

    results = []
    for ts, img in frames:
        raw = caption_one(blip_processor, blip_model, img)
        if not raw:
            continue
        # Domain check on raw caption
        if is_cricket_visual(raw):
            context_cap = add_cricket_context(raw)
            results.append((ts, context_cap))
            print(f"  [{int(ts//60):02d}:{int(ts%60):02d}] ✅ {context_cap}")
        else:
            print(f"  [{int(ts//60):02d}:{int(ts%60):02d}] ❌ Non-cricket frame skipped: {raw}")

    print(f"  [Video captions] {len(results)}/{len(frames)} frames accepted")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Image file captioning
# ─────────────────────────────────────────────────────────────────────────────

def caption_image_file(image_path: str, blip_processor, blip_model, index: int = 0):
  
    if blip_model is None or blip_processor is None:
        print(f"  [BLIP] Model not loaded — image #{index} skipped")
        return None, None

    try:
        print(f"  [BLIP] Processing image: {image_path}")
        img = Image.open(image_path)

        # FIX: Convert any mode to RGB safely
        if img.mode != "RGB":
            print(f"    [Image] Converting {img.mode} → RGB")
            img = img.convert("RGB")

        print(f"    [Image] Size: {img.size}, mode: {img.mode}")

        raw = caption_one(blip_processor, blip_model, img)
        if not raw:
            print(f"    [BLIP] No usable caption generated for image #{index}")
            return None, None

        context_cap = add_cricket_context(raw)
        print(f"    [BLIP] Raw: {repr(raw)}")
        print(f"    [BLIP] Context: {repr(context_cap)}")
        return raw, context_cap

    except FileNotFoundError:
        print(f"  [Image ERROR] File not found: {image_path}")
        return None, None
    except Exception as e:
        print(f"  [Image ERROR] {type(e).__name__}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_video_captions(captions) -> str:
    if not captions:
        return ""
    return "\n".join(
        f"[{int(ts//60):02d}:{int(ts%60):02d}] {cap}"
        for ts, cap in captions
    )
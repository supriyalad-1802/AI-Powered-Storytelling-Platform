import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def assess_data_sufficiency(transcript, text_contents, video_captions, image_captions):
    """Evaluate if the extracted data is sufficient to generate a coherent story."""
    # Basic word count from transcript and texts
    total_words = len(transcript.split()) + sum(len(c.split()) for _, c in text_contents)
    
    if total_words < 50 and not video_captions and not image_captions:
        return {
            "sufficient": False, 
            "total_words": total_words,
            "suggestion": "Please upload more commentary audio, video with speech, or text scorecards."
        }
    return {"sufficient": True, "total_words": total_words, "suggestion": ""}

def build_fused_document(transcript, video_captions, image_captions, text_contents, entities):
    """Combine all extracted multimodal data into a single formatted string for the LLM."""
    fused = ""
    if transcript:
        fused += f"--- AUDIO TRANSCRIPT ---\n{transcript}\n\n"
    if text_contents:
        fused += "--- TEXT DOCUMENTS ---\n"
        for name, content in text_contents:
            fused += f"[{name}]: {content}\n\n"
    if video_captions:
        fused += "--- VIDEO VISUALS ---\n"
        for ts, cap in video_captions:
            fused += f"[{ts}]: {cap}\n"
        fused += "\n"
    if image_captions:
        fused += "--- IMAGE VISUALS ---\n"
        for name, cap in image_captions:
            fused += f"[{name}]: {cap}\n"
        fused += "\n"
    return fused.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Story Generator (LLM Brain)
# ─────────────────────────────────────────────────────────────────────────────
def generate_story(fused_doc, transcript, video_captions, image_captions,
                   text_contents, entities, event_name, target_minutes=2, language="English", custom_instruction=""):
    """Call the LLM to generate the sports narrative in the requested language, with SSML tags and highlight timestamp."""
                   
    # 1. Check if we have absolute minimum data
    suf = assess_data_sufficiency(transcript, text_contents, video_captions, image_captions)
    if not suf["sufficient"]:
        return None, suf["suggestion"]

    total_input_words = suf["total_words"]
    
    # 2. THE STRICT LENGTH CONTROLLER
    # Edge TTS reads at roughly 150 words per minute
    words_per_minute = 150
    requested_words = target_minutes * words_per_minute
    
    # Only shrink the story if the data is so small that it would cause total hallucination
    max_safe_words = total_input_words * 4 
    
    if requested_words > max_safe_words and total_input_words < 100:
        print("⚠️ Data exceptionally short. Auto-adjusting to fit facts.")
        target_words = max(max_safe_words, 150) 
    else:
        # Enforce the strict requested time
        target_words = requested_words

    print(f"\n🔹 Requesting {target_minutes}-minute {language} story from Groq LLM (Target Words: ~{target_words})...")

    # 3. Define the LLM's Persona and Strict Expanding Rules
    script_instruction = "using ONLY the Devanagari script (हिंदी). DO NOT use Romanized Hindi/Hinglish. DO NOT include any English translations." if language == "Hindi" else "DO NOT include any Hindi translations."
    
    video_splicing_rule = ""


    system_prompt = (
        f"You are an elite sports television commentator writing a thrilling cricket match feature.\n"
        f"CRITICAL LANGUAGE RULE: You MUST write the ENTIRE story, including the headline, strictly and exclusively in fluent {language} {script_instruction}.\n"
        f"CRITICAL LENGTH RULE: You MUST write a script of EXACTLY {target_words} words to perfectly fill a {target_minutes}-minute television broadcast segment. This is a non-negotiable hard limit.\n"
        "EXPANSION TACTICS: Do NOT invent fake match statistics, runs, specific ball-by-ball events, or specific player names that are not explicitly present in the provided data. To expand the narrative, describe the general atmosphere, the psychological tension of the players as a whole, or praise the overall performance of the team.\n"
        "CRITICAL FORMAT RULE: DO NOT use any Markdown formatting. NO asterisks (*), NO bolding, NO hashtags. Write purely in plain text.\n"
        "BROADCAST RULES FOR AUDIO DYNAMICS:\n"
        "- Start the story IMMEDIATELY with a bold, exciting headline at the very top (in plain text, ALL CAPS). DO NOT prepend 'Event Name:' or any introductory text.\n"
        "- Use ellipses (...) heavily for dramatic pauses.\n"
        "- Use exclamation marks (!) for huge moments.\n"
        "SSML INJECTION: You must insert valid SSML tags directly into the text for emotional impact:\n"
        '- Use <prosody rate="+20%" pitch="+10Hz"> around high-action moments (wickets, sixes) to simulate shouting.</prosody>\n'
        '- Use <prosody rate="-10%"> around tense, dramatic build-ups.</prosody>\n'
        '- Insert <break time="800ms"/> before revealing match results or umpire decisions.\n'
        f"{video_splicing_rule}"
    )

    # Inject the user's custom chatbot-style prompt if they provided one
    custom_directive = ""
    if custom_instruction:
        custom_directive = f"\n\nUSER'S SPECIAL INSTRUCTIONS: {custom_instruction}\n(You must follow these instructions closely while writing the narrative.)"

    user_prompt = f"Here is the extracted data from the user's uploads for the match '{event_name}':\n\n{fused_doc}{custom_directive}\n\nPlease write the final sports story."

    # 4. Call the Groq API
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.75, 
            max_tokens=2500
        )
        
        narrative = response.choices[0].message.content
        return narrative, None

    except Exception as e:
        print(f"❌ LLM Generation Error: {e}")
        return None, f"Story generation failed. Error: {str(e)}"
# The fallback function that was missing!
def build_template_narrative(transcript, video_captions, image_captions, text_contents, entities, event_name):
    """Fallback narrative generator if the LLM call fails."""
    return f"MATCH REPORT: {event_name}\n\nAn automated story could not be generated due to an API error, but the match data was successfully processed. Please check your console logs to debug."
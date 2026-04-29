import os
import asyncio
import edge_tts

def text_to_audio_file(text, output_filepath, language="English"):
    """Synthesize text (or SSML) to an audio file using edge-tts based on the chosen language."""
    
    if language == "Hindi":
        voice = "hi-IN-MadhurNeural"
    else:
        voice = "en-US-ChristopherNeural"
    
    async def _generate():
        # Edge-tts Communicate will wrap the text in <voice>. If the text contains valid SSML tags
        # like <prosody> or <break>, they will be parsed by the Edge TTS engine.
        # We removed the global rate="+5%" to prevent edge-tts from double-wrapping the text in
        # a <prosody> tag, which might conflict with our custom injected SSML tags.
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_filepath)
        
    try:
        # Flask runs synchronously, so we safely spin up an async event loop here
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate())
        loop.close()
        
        # Verify the file was actually created
        if os.path.exists(output_filepath):
            return output_filepath, None
        else:
            return None, "Audio file was not created successfully."
            
    except Exception as e:
        return None, f"TTS Engine Error: {str(e)}"
# Multimodal AI Sports Storytelling Pipeline 🏏

An advanced, edge-to-cloud AI orchestration pipeline that takes unstructured sports media (videos, audio, images, and text) and synthesizes them into dynamic, broadcast-quality sports narratives in both English and Hindi. 

## Features

- **Multimodal Data Extraction:** Automatically processes MP4s, WAVs, JPEGs, and text files. Uses **OpenAI Whisper** for transcription and **Salesforce BLIP** for visual frame captioning.
- **LLM Orchestration:** Powered by the **Groq Llama-3.3-70b-versatile** model to fuse extracted metadata into an engaging, strictly timed television broadcast narrative.
- **Multilingual Broadcasting:** Select between English or Hindi outputs. Automatically maps to native Azure Edge-TTS voices for high-quality audio.
- **Expressive SSML Injection:** Leverages Speech Synthesis Markup Language (SSML) to dynamically add dramatic pauses and emotional pitch shifts to the generated audio narration.
- **Automated Video Highlights:** Identifies the story's climax and uses `moviepy` to automatically splice and present a 10-second highlight reel on the UI.
- **Premium Interface:** A fully responsive, glassmorphism-inspired Tailwind UI with a live typewriter effect and real-time audio word-highlighting.

---

## Architecture Overview

- **Backend:** Python, Flask, asyncio
- **Frontend:** Vanilla JS, HTML, Tailwind CSS
- **Local AI (Edge):** OpenAI Whisper (audio), Salesforce BLIP (vision)
- **Cloud AI:** Groq API (narrative), Microsoft Edge-TTS (audio synthesis)

---

## Getting Started

Follow these steps to run the pipeline locally.

### 1. Prerequisites

You **must** have FFmpeg installed on your system for audio/video processing:
- **Windows:** `choco install ffmpeg` (or download from [ffmpeg.org](https://ffmpeg.org))
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### 2. Environment Setup

Clone the repository and navigate to the project directory:

```bash
# Recommended: Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages from the requirements file located in the frontend folder:

```bash
pip install -r frontend/req.txt
```

### 4. Environment Variables

Create a `.env` file in the root directory (or in the `backend/` folder) and add your Groq API key:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 5. Run the Application

Start the Flask server from the root directory:

```bash
python backend/app.py
```

The application will start a local server. Open your web browser and navigate to:
**http://127.0.0.1:5000**

---

## How to Use

1. **Upload Media:** Drag and drop cricket match videos, commentary audio, scorecard text, or match photos.
2. **Configure Output:** Enter the match name, target audio length, optional custom director instructions, and select the output language (English or Hindi).
3. **Execute:** Click "Initialize Pipeline". The system will ingest the data, process the AI models, and render the final story.
4. **Playback:** If Auto-play is enabled, the expressive audio will immediately synthesize and play, synchronizing with the text on screen. Any generated highlight video will loop automatically above the text.

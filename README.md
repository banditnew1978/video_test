# 🎬 AI Video Dubbing & Lip-Sync Pipeline

This project is a fully automated, end-to-end video dubbing pipeline. It extracts vocals, transcribes and diarizes speakers, translates the text contextually using local LLMs, automatically assigns gender-matched high-quality voices (or preserves the original speaker's voice), generates TTS audio with optional **tone color transfer** from the original speaker, perfectly time-stretches the audio to match the original lip movements, and finally merges everything back into a seamless dubbed video.

## ✨ Key Features

*   **🎙️ Speaker Diarization:** Uses OpenAI Whisper & NeMo to accurately map "who spoke when".
*   **🧠 Contextual AI Translation:** Connects to local LLMs (via LM Studio) to translate dialogs naturally without losing context.
*   **⏱️ Dynamic Lip-Sync (Time Stretching):** Uses FFmpeg `atempo` to stretch (slow down) or squeeze (speed up) the generated TTS audio to perfectly fit the original speaker's exact speaking window.
*   **🧑‍🦱 Voice Gender Classification:** Automatically analyzes speaker audio to detect gender. You can choose to keep the **original speaker's voice** or assign a predefined reference voice from your library.
*   **🎨 Tone Color Conversion (OpenVoice):** When using a custom voice, optionally transfer the original speaker's unique tone/timbre onto the TTS output — giving you the expressiveness of your custom voice with the identity of the original speaker.
*   **🎵 Background Audio Preservation:** Extracts and preserves background music/noise (via Demucs) and mixes it under the new dubbed voices.

---

## 📦 Distrobox Installation (Recommended)

This is the recommended way to run the pipeline to avoid dependency conflicts and keep your host system clean.

**1. Create Home Directory on SSD:**
```bash
mkdir -p /mnt/usbssd/Distrobox_home/video_dubbing_2404
```

**2. Create the Container:**
```bash
distrobox create --name video-dubbing-2404 \
                 --image ubuntu:24.04 \
                 --home /mnt/usbssd/Distrobox_home/video_dubbing_2404 \
                 --nvidia --yes
```

**3. Install System Dependencies (Inside Container):**
```bash
distrobox enter video-dubbing-2404 -- sudo apt update
distrobox enter video-dubbing-2404 -- sudo apt install -y python3-pip python3-venv ffmpeg git libsndfile1 build-essential libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-shape0 libgl1
```

**4. Setup Python Environment:**
```bash
distrobox enter video-dubbing-2404
cd /mnt/depo_hdd/video_dubbing_Docker
python3 -m venv .venv_2404
source .venv_2404/bin/activate

# Install requirements in the correct order
pip install -r requirements_env.txt
pip install -r requirements.txt -c constraints.txt
pip install git+https://github.com/griko/voice-gender-classification.git
```

---

## 🛠️ Native Installation (Host)

---

## ⚙️ Setup & Preparation

### 1. Local Translation (LM Studio)
If you are using a local LLM for translations, start your LM Studio local server and ensure it is running on `http://localhost:1234`.

### 2. Custom Voice References
Place your high-quality reference audio files and their matching text transcripts in the `voice/` directory. Name them clearly (e.g., include "male" or "female" in the filename).
*   `voice/male1.wav` & `voice/male1.txt`
*   `voice/female1.wav` & `voice/female1.txt`

---

## 🚀 Usage (`claude_main.py`)

The heart of the pipeline is `claude_main.py`. This script orchestrates all the underlying tools (`diarize.py`, `helpers.py`, OmniVoice, FFmpeg) into one cohesive workflow.

### Basic Interactive Run
```bash
python claude_main.py --video my_movie.mp4 \
                      --target_language Turkish \
                      --target_language_id tr \
                      --lm_studio_url http://localhost:1234
```

### 💬 Interactive Prompts Explained
When you run the script, it will pause at key moments to ask for your input:
1.  **Translation Strategy:** Do you want the AI to summarize/compress the text to fit the original time, or do you want a natural translation (and let the pipeline physically speed up the audio later)?
2.  **Silence Trimming:** Do you want aggressive silence trimming (-40dB) for strict lip-sync accuracy?
3.  **Voice Assignment (Step 3.5):** After diarizing the video, the pipeline will detect each speaker's gender. For each speaker, you can choose:
    *   `[0]` 🎬 **Keep Original Voice** — Use the speaker's own voice extracted from the video.
    *   `[1..N]` 🎙️ **Custom Voice** — Pick a high-quality reference voice from your `voice/` folder.
4.  **Tone Color Conversion:** If you select a custom voice, the pipeline asks whether to apply OpenVoice tone conversion. This transfers the original speaker's unique timbre onto the custom-voiced TTS output — the custom voice provides speech style and language quality, while the tone matches the original speaker.

### 🤖 Unattended / Automation Mode
If you are running batch jobs and want to skip all terminal questions, use the `--no_confirm` flag. The system will automatically pick the first available voice that matches the speaker's detected gender.
```bash
python claude_main.py --video my_movie.mp4 --target_language Turkish --target_language_id tr --no_confirm
```

---

## 🎛️ Advanced Arguments

You can heavily customize the pipeline's behavior using the following arguments:

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--translation_api` | String | `lm_studio` | Which API to use (kept for backward compatibility). |
| `--max_speed_factor` | Float | `1.25` | Max FFmpeg speed-up ratio before falling back to OmniVoice forced duration. |
| `--max_shortening_ratio` | Float | `0.40` | Max slow-down (stretching) limit. |
| `--chars_per_second` | Float | `21.0` | TTS chars/second calibration — controls how long the translated text should be. Increase if TTS audio is too short, decrease if too long. |
| `--vocal_volume` | Float | `0.9` | Volume level of the generated TTS voices. |
| `--bg_volume` | Float | `1.0` | Volume level of the original background audio. |
| `--silence_threshold` | String | `-40dB` | Silence trimming threshold for TTS outputs. |
| `--keep_tmp` | Flag | `False` | Prevent the script from deleting the `[video_name]_tmp` folder after completion (useful for debugging). |
| `--no_confirm` | Flag | `False` | Skip all interactive prompts (uses original voices by default). |

---

## 📂 File Structure

*   `claude_main.py`: The main orchestrator script. Run this.
*   `gui.py`: PyQt6 GUI for easy pipeline configuration and execution.
*   `diarize.py`: Handled by `claude_main.py` in the background for speaker separation and Whisper transcription.
*   `helpers.py`: Utility functions used by `diarize.py`.
*   `voice/`: Directory for storing your custom `.wav` and `.txt` reference voices.

---

## 🔄 Pipeline Flow

```
Step 1  → Demucs: Separate vocals from background audio
Step 2  → Whisper + NeMo: Transcribe & diarize speakers
Step 3  → Extract reference audio per speaker (3-10s clips)
Step 3.5 → Gender detection → Voice selection (original or custom)
           → Optional: Tone color conversion preference
Step 4  → LM Studio: Contextual translation with char budget
Step 6  → OmniVoice TTS: Generate speech with selected voice
Step 6.5 → OpenVoice: Tone color transfer (if enabled)
Step 7  → FFmpeg atempo: Speed adjustment for lip-sync
Step 8  → FFmpeg: Assemble final dubbed video
Step 10 → Generate pipeline report (JSON)
```

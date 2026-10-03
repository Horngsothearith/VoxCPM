# 🎙️ VoxCPM Local API Server & Playground

A high-performance local REST & Streaming API for [VoxCPM / VoxCPM2](https://github.com/OpenBMB/VoxCPM) featuring **OpenAI `/v1/audio/speech` compatibility**, **multilingual TTS (30+ languages)**, **natural language Voice Design**, and **true-to-life Voice Cloning**.

---

## ⚡ Quick Start

### 1. Launch the API Server

You can start the server with one command:

#### Option A: Docker Compose (Recommended)
```bash
docker compose up -d
```
To view logs:
```bash
docker compose logs -f
```
To stop:
```bash
docker compose down
```

#### Option B: Windows Batch (Double-click or run from CMD)
```cmd
run_api.bat
```

#### Option C: PowerShell
```powershell
.\run_api.ps1
```

#### Option D: Python directly
```bash
# Using virtual environment
.\.venv\Scripts\python.exe api_server.py --port 8000 --device auto
```

### 2. Open the Interactive Web Playground
Open your browser and navigate to:
👉 **[http://localhost:8000/ui](http://localhost:8000/ui)** (or simply `http://localhost:8000/`)

Interactive Swagger API docs are available at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status, model info, device, GPU status |
| `GET` | `/v1/models` | List available models (OpenAI compatible) |
| `GET` | `/v1/tts/voices` | List built-in voice presets and descriptions |
| `POST` | `/v1/audio/speech` | **OpenAI-compatible TTS API** (works with OpenWebUI, LibreChat, etc.) |
| `POST` | `/v1/tts/generate` | Full VoxCPM generation endpoint (Voice Design & JSON audio) |
| `POST` | `/v1/tts/clone` | Multipart file upload endpoint for zero-shot Voice Cloning |
| `POST` | `/v1/tts/stream` | Low-latency streaming TTS (raw 16-bit PCM chunks) |
| `GET` | `/ui` | Sleek dark-mode Web UI Playground |

---

## 🚀 Usage Examples

### 1. OpenAI-Compatible Endpoint (`POST /v1/audio/speech`)

This endpoint is drop-in compatible with any software expecting OpenAI's TTS API:

#### cURL
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "Hello! VoxCPM is now running locally on your computer.",
    "voice": "nova",
    "response_format": "wav"
  }' \
  --output speech.wav
```

#### Python (`openai` SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="voxcpm2",
    voice="shimmer",  # preset: alloy, echo, fable, onyx, nova, shimmer
    input="The rain in Spain stays mainly in the plain."
)
response.stream_to_file("speech.wav")
```

> **Tip:** You can also pass a custom voice description directly in the `voice` parameter:
> ```python
> response = client.audio.speech.create(
>     model="voxcpm2",
>     voice="(A cheerful young woman with a sweet and gentle voice)",
>     input="Welcome to the future of voice synthesis!"
> )
> ```

---

### 2. Creative Voice Design (`POST /v1/tts/generate`)

Create completely new voices simply by describing them:

#### cURL
```bash
curl -X POST http://localhost:8000/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The universe is under no obligation to make sense to you.",
    "voice_design": "Deep, mature documentary narrator, calm and dramatic pauses",
    "cfg_value": 2.0,
    "inference_timesteps": 10,
    "response_format": "wav"
  }' \
  --output narrator.wav
```

---

### 3. Voice Cloning with Audio Upload (`POST /v1/tts/clone`)

Upload a 5–15 second `.wav` or `.mp3` reference clip:

#### cURL
```bash
curl -X POST http://localhost:8000/v1/tts/clone \
  -F "file=@my_voice.wav" \
  -F "text=This is my voice cloned locally by VoxCPM." \
  -F "control=slightly faster, cheerful tone" \
  --output cloned_voice.wav
```

#### Python (`requests`)
```python
import requests

with open("my_voice.wav", "rb") as f:
    files = {"file": f}
    data = {
        "text": "This voice was synthesized directly from my microphone sample.",
        "control": "warm and friendly"
    }
    res = requests.post("http://localhost:8000/v1/tts/clone", files=files, data=data)

with open("cloned.wav", "wb") as out:
    out.write(res.content)
```

---

## ⚙️ Configuration & Flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | IP to bind to (`0.0.0.0` allows access from local network) |
| `--port` | `8000` | Port number |
| `--device` | `auto` | Runtime device: `auto`, `cpu`, `cuda`, `cuda:0` |
| `--model-id` | `openbmb/VoxCPM2` | Model repository or local directory path |
| `--preload` | `false` | Load weights into memory immediately at startup |
| `--load-denoiser` | `false` | Load ZipEnhancer denoiser pipeline |
| `--optimize` | `false` | Enable `torch.compile` (recommended only on Linux with CUDA) |

---

## 🔌 Connecting to Third-Party Apps

### OpenWebUI / LibreChat
1. In Settings ➔ Audio ➔ Text-to-Speech:
   - **TTS Engine**: `OpenAI`
   - **API Base URL**: `http://localhost:8000/v1`
   - **API Key**: `sk-local` (any string)
   - **TTS Model**: `voxcpm2`
   - **Voice**: `nova` (or `alloy`, `echo`, `shimmer`, etc.)

"""
VoxCPM FastAPI Server
=====================
High-performance, OpenAI-compatible local API server for VoxCPM / VoxCPM2.

Features:
- OpenAI-compatible TTS endpoint: POST /v1/audio/speech
- Native VoxCPM generation endpoint: POST /v1/tts/generate
- Voice Cloning with file upload: POST /v1/tts/clone
- Streaming audio endpoint: POST /v1/tts/stream
- Health & Model Info: GET /health, GET /v1/models, GET /v1/tts/voices
- Interactive Web UI Playground: GET /ui (and GET /)
"""

import os
import sys
import io
import time
import base64
import logging
import argparse
import tempfile
import threading
from typing import Optional, List, Dict, Any, Generator
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Request, Response
from fastapi.responses import Response, StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Ensure src is in python path
workspace_dir = Path(__file__).parent.resolve()
src_dir = workspace_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import voxcpm
from voxcpm import VoxCPM
from voxcpm.model.utils import resolve_runtime_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxcpm_api")

# Global configuration and model instance
class ServerState:
    model: Optional[VoxCPM] = None
    model_id: str = "openbmb/VoxCPM2"
    device: str = "auto"
    resolved_device: str = "cpu"
    sample_rate: int = 48000
    load_denoiser: bool = False
    optimize: bool = False
    model_lock: threading.Lock = threading.Lock()
    start_time: float = time.time()
    is_loading: bool = False

state = ServerState()

# Predefined Voice Descriptions (OpenAI voice mapping & presets)
VOICE_PRESETS: Dict[str, Dict[str, str]] = {
    "alloy": {
        "name": "Alloy",
        "gender": "neutral",
        "description": "(A balanced, versatile neutral voice with clear articulation and steady cadence)"
    },
    "echo": {
        "name": "Echo",
        "gender": "male",
        "description": "(A warm, deep, resonant male voice speaking in a calm, confident tone)"
    },
    "fable": {
        "name": "Fable",
        "gender": "male",
        "description": "(An expressive, warm British-accented storytelling voice with nuanced intonation)"
    },
    "onyx": {
        "name": "Onyx",
        "gender": "male",
        "description": "(A deep, authoritative baritone male voice, formal and grounded)"
    },
    "nova": {
        "name": "Nova",
        "gender": "female",
        "description": "(A bright, cheerful, energetic young female voice with an upbeat attitude)"
    },
    "shimmer": {
        "name": "Shimmer",
        "gender": "female",
        "description": "(A gentle, soft, melodic and sweet young female voice, speaking warmly)"
    },
    "storyteller": {
        "name": "Storyteller",
        "gender": "male",
        "description": "(An engaging documentary narrator, mature, rich tone with natural dramatic pauses)"
    },
    "coach": {
        "name": "Energetic Coach",
        "gender": "male",
        "description": "(High energy, motivational, fast-paced coach voice filled with enthusiasm)"
    },
    "whisper": {
        "name": "Soft Whisper",
        "gender": "female",
        "description": "(A gentle, breathy, intimate whispering female voice, very soft and calm)"
    },
    "news": {
        "name": "News Anchor",
        "gender": "female",
        "description": "(Professional broadcast news anchor, crisp, neutral, clear, articulate)"
    }
}

def get_or_load_model() -> VoxCPM:
    """Lazy-load or return existing VoxCPM model instance in a thread-safe manner."""
    if state.model is not None:
        return state.model

    with state.model_lock:
        if state.model is not None:
            return state.model

        state.is_loading = True
        logger.info(f"Loading VoxCPM model '{state.model_id}' on device '{state.device}'...")
        try:
            # Check CUDA availability
            if state.device == "auto":
                state.resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                state.resolved_device = state.device

            logger.info(f"Resolved execution device: {state.resolved_device}")
            
            # Windows torch.compile optimization safe-guard
            opt = state.optimize
            if sys.platform == "win32" and opt:
                logger.warning("torch.compile optimization disabled on Windows to prevent C++ compiler runtime errors")
                opt = False

            model = VoxCPM.from_pretrained(
                hf_model_id=state.model_id,
                load_denoiser=state.load_denoiser,
                optimize=opt,
                device=state.resolved_device,
            )
            state.model = model
            state.sample_rate = getattr(model.tts_model, "sample_rate", 48000)
            logger.info(f"VoxCPM Model loaded successfully! Sample rate: {state.sample_rate} Hz")
            return model
        except Exception as e:
            logger.error(f"Failed to load VoxCPM model: {e}", exc_info=True)
            raise RuntimeError(f"Model initialization failed: {str(e)}")
        finally:
            state.is_loading = False

def audio_numpy_to_wav_bytes(audio_arr: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy 1D float32 audio array to WAV bytes (PCM_16)."""
    buf = io.BytesIO()
    # Normalize if needed
    max_val = np.max(np.abs(audio_arr))
    if max_val > 1.0:
        audio_arr = audio_arr / max_val
    sf.write(buf, audio_arr, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# ----------------- Request / Response Models -----------------

class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="voxcpm2", description="Model name (e.g. voxcpm2)")
    input: str = Field(..., description="The text to generate audio for")
    voice: Optional[str] = Field(default="alloy", description="Voice preset name (alloy, echo, nova, etc.) or custom prompt '(A young woman...)'")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav, pcm)")
    speed: Optional[float] = Field(default=1.0, description="Speech speed adjustment (hint in voice design)")

class TTSGenerateRequest(BaseModel):
    text: str = Field(..., description="Target text to synthesize")
    voice_design: Optional[str] = Field(default=None, description="Natural language voice description (e.g. 'A cheerful young woman...')")
    reference_audio_base64: Optional[str] = Field(default=None, description="Base64 encoded reference audio for voice cloning")
    prompt_audio_base64: Optional[str] = Field(default=None, description="Base64 encoded prompt audio for ultimate continuation cloning")
    prompt_text: Optional[str] = Field(default=None, description="Transcript of prompt audio (required with prompt_audio)")
    cfg_value: float = Field(default=2.0, ge=1.0, le=10.0, description="Classifier-Free Guidance value")
    inference_timesteps: int = Field(default=10, ge=1, le=50, description="Diffusion steps (default 10)")
    seed: Optional[int] = Field(default=None, description="Random seed for deterministic generation")
    normalize: bool = Field(default=False, description="Run text normalization")
    denoise: bool = Field(default=False, description="Denoise input reference audio")
    response_format: str = Field(default="wav", description="Output format: 'wav' (binary audio) or 'json' (base64 audio + metadata)")

# ----------------- FastAPI App Initialization -----------------

app = FastAPI(
    title="VoxCPM Local API",
    description="Local REST & Streaming API for VoxCPM2: Tokenizer-Free Multilingual TTS, Creative Voice Design, and High-Fidelity Voice Cloning.",
    version="2.0.0",
)

# CORS middleware for Web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Endpoints -----------------

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint returning server status and runtime device."""
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else None
    
    return {
        "status": "online",
        "model_id": state.model_id,
        "is_model_loaded": state.model is not None,
        "is_loading": state.is_loading,
        "device": state.resolved_device if state.model is not None else (
            ("cuda" if has_gpu else "cpu") if state.device == "auto" else state.device
        ),
        "cuda_available": has_gpu,
        "gpu_name": gpu_name,
        "sample_rate": state.sample_rate,
        "uptime_seconds": round(time.time() - state.start_time, 2)
    }

@app.get("/v1/models", tags=["OpenAI Compatible"])
async def list_models():
    """OpenAI-compatible models list endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "voxcpm2",
                "object": "model",
                "created": int(state.start_time),
                "owned_by": "openbmb",
                "permission": [],
                "root": "voxcpm2",
                "parent": None
            },
            {
                "id": "openbmb/VoxCPM2",
                "object": "model",
                "created": int(state.start_time),
                "owned_by": "openbmb",
                "permission": [],
                "root": "voxcpm2",
                "parent": None
            }
        ]
    }

@app.get("/v1/tts/voices", tags=["VoxCPM"])
async def list_voices():
    """List built-in voice design presets."""
    return {
        "voices": [
            {"id": key, **val}
            for key, val in VOICE_PRESETS.items()
        ]
    }

@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech(req: OpenAISpeechRequest):
    """
    OpenAI-compatible Text-To-Speech endpoint.
    Compatible with OpenAI clients, OpenWebUI, LibreChat, SillyTavern, etc.
    """
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    model = await run_in_threadpool(get_or_load_model)
    
    # Process voice description
    voice_key = (req.voice or "alloy").lower().strip()
    full_text = req.input.strip()

    # If the voice matches a preset, prepend description
    if voice_key in VOICE_PRESETS:
        preset_desc = VOICE_PRESETS[voice_key]["description"]
        if not full_text.startswith("("):
            full_text = f"{preset_desc}{full_text}"
    elif voice_key.startswith("(") and ")" in voice_key:
        # User passed a custom voice design in the voice parameter
        full_text = f"{voice_key}{full_text}"
    elif voice_key not in ["default", "none", ""]:
        # Treat unknown voice string as voice description hint
        full_text = f"({voice_key}){full_text}"

    # Handle speed hint if significantly different from 1.0
    if req.speed and abs(req.speed - 1.0) > 0.15:
        speed_hint = "faster pace, " if req.speed > 1.0 else "slower pace, "
        if full_text.startswith("("):
            idx = full_text.find(")")
            full_text = f"({speed_hint}{full_text[1:idx]}){full_text[idx+1:]}"
        else:
            full_text = f"({speed_hint}){full_text}"

    try:
        def _run():
            with state.model_lock:
                return model.generate(
                    text=full_text,
                    cfg_value=2.0,
                    inference_timesteps=10,
                )
        wav = await run_in_threadpool(_run)
            
        wav_bytes = audio_numpy_to_wav_bytes(wav, state.sample_rate)
        
        # Raw PCM option
        if req.response_format == "pcm":
            pcm_data = (wav * 32767).astype(np.int16).tobytes()
            return Response(content=pcm_data, media_type="audio/pcm")
            
        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Error during OpenAI speech generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tts/generate", tags=["VoxCPM"])
async def tts_generate(req: TTSGenerateRequest):
    """
    Full-featured VoxCPM generation endpoint.
    Supports Voice Design, Controllable Voice Cloning, and Ultimate Cloning.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model = await run_in_threadpool(get_or_load_model)
    
    # Format text with voice design if provided
    final_text = req.text.strip()
    if req.voice_design and req.voice_design.strip():
        design = req.voice_design.strip()
        if not design.startswith("("):
            design = f"({design})"
        final_text = f"{design}{final_text}"

    temp_files = []
    ref_path = None
    prompt_path = None

    try:
        # Handle reference audio from base64
        if req.reference_audio_base64:
            ref_bytes = base64.b64decode(req.reference_audio_base64)
            ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            ref_file.write(ref_bytes)
            ref_file.close()
            temp_files.append(ref_file.name)
            ref_path = ref_file.name

        # Handle prompt audio from base64 (Ultimate Continuation Mode)
        if req.prompt_audio_base64:
            prompt_bytes = base64.b64decode(req.prompt_audio_base64)
            prompt_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            prompt_file.write(prompt_bytes)
            prompt_file.close()
            temp_files.append(prompt_file.name)
            prompt_path = prompt_file.name

        t0 = time.time()
        def _run():
            with state.model_lock:
                return model.generate(
                    text=final_text,
                    prompt_wav_path=prompt_path,
                    prompt_text=req.prompt_text,
                    reference_wav_path=ref_path,
                    cfg_value=req.cfg_value,
                    inference_timesteps=req.inference_timesteps,
                    normalize=req.normalize,
                    denoise=req.denoise,
                    seed=req.seed,
                )
        wav = await run_in_threadpool(_run)
        elapsed = time.time() - t0
        duration = len(wav) / state.sample_rate

        wav_bytes = audio_numpy_to_wav_bytes(wav, state.sample_rate)

        if req.response_format == "json":
            return {
                "sample_rate": state.sample_rate,
                "duration_seconds": round(duration, 3),
                "generation_time_seconds": round(elapsed, 3),
                "rtf": round(elapsed / max(duration, 0.001), 3),
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "format": "wav"
            }

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(round(duration, 3)),
                "X-Generation-Time": str(round(elapsed, 3)),
                "X-Sample-Rate": str(state.sample_rate),
            }
        )

    except Exception as e:
        logger.error(f"Error during TTS generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in temp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

@app.post("/v1/tts/clone", tags=["VoxCPM"])
async def tts_clone_file(
    file: UploadFile = File(..., description="Reference audio file (.wav, .mp3, etc.)"),
    text: str = Form(..., description="Target text to synthesize"),
    control: Optional[str] = Form(None, description="Optional style/speed/emotion control (e.g. 'cheerful, slightly faster')"),
    prompt_text: Optional[str] = Form(None, description="If provided, activates Ultimate Cloning continuation mode"),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    seed: Optional[int] = Form(None),
):
    """
    Direct multipart file-upload endpoint for voice cloning.
    Upload any reference audio clip and synthesize target text.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model = await run_in_threadpool(get_or_load_model)
    
    # Save uploaded audio file to temporary location
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    temp_ref = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    contents = await file.read()
    temp_ref.write(contents)
    temp_ref.close()

    final_text = text.strip()
    if control and control.strip():
        ctrl = control.strip()
        if not ctrl.startswith("("):
            ctrl = f"({ctrl})"
        final_text = f"{ctrl}{final_text}"

    prompt_path = temp_ref.name if prompt_text else None

    try:
        t0 = time.time()
        def _run():
            with state.model_lock:
                return model.generate(
                    text=final_text,
                    prompt_wav_path=prompt_path,
                    prompt_text=prompt_text,
                    reference_wav_path=temp_ref.name,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    seed=seed,
                )
        wav = await run_in_threadpool(_run)
        elapsed = time.time() - t0
        duration = len(wav) / state.sample_rate
        wav_bytes = audio_numpy_to_wav_bytes(wav, state.sample_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(round(duration, 3)),
                "X-Generation-Time": str(round(elapsed, 3)),
                "X-Sample-Rate": str(state.sample_rate),
            }
        )
    except Exception as e:
        logger.error(f"Error during voice cloning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if os.path.exists(temp_ref.name):
                os.remove(temp_ref.name)
        except Exception:
            pass

@app.post("/v1/tts/stream", tags=["VoxCPM"])
async def tts_stream(req: TTSGenerateRequest):
    """
    Real-time streaming TTS endpoint.
    Streams raw 16-bit PCM audio chunks (48kHz mono) with low time-to-first-audio.
    """
    model = await run_in_threadpool(get_or_load_model)
    
    final_text = req.text.strip()
    if req.voice_design and req.voice_design.strip():
        design = req.voice_design.strip()
        if not design.startswith("("):
            design = f"({design})"
        final_text = f"{design}{final_text}"

    def stream_generator():
        with state.model_lock:
            for chunk in model.generate_streaming(
                text=final_text,
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
                seed=req.seed,
            ):
                # Convert chunk to 16-bit PCM
                pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
                yield pcm_bytes

    return StreamingResponse(stream_generator(), media_type="audio/pcm")

# ----------------- Interactive Web UI Playground -----------------

@app.get("/", response_class=HTMLResponse, tags=["Playground"])
@app.get("/ui", response_class=HTMLResponse, tags=["Playground"])
async def web_playground():
    """Interactive, sleek dark-mode Web UI Playground for testing VoxCPM API."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VoxCPM2 Local API & Playground</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 29, 47, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-primary: #6366f1;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --accent-glow: rgba(99, 102, 241, 0.3);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --success: #10b981;
      --danger: #ef4444;
      --radius: 14px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 50% 0%, #171d34 0%, var(--bg) 70%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px;
    }

    .container {
      width: 100%;
      max-width: 960px;
    }

    header {
      text-align: center;
      margin-bottom: 28px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      color: #818cf8;
      margin-bottom: 12px;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 10px var(--success);
    }

    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
      margin-bottom: 6px;
    }

    p.subtitle {
      color: var(--text-muted);
      font-size: 1.05rem;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }

    .tabs {
      display: flex;
      gap: 8px;
      background: rgba(11, 15, 25, 0.6);
      padding: 6px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      margin-bottom: 20px;
    }

    .tab-btn {
      flex: 1;
      padding: 10px 16px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 500;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .tab-btn.active {
      background: var(--accent-primary);
      color: white;
      box-shadow: 0 4px 14px var(--accent-glow);
    }

    .form-group {
      margin-bottom: 18px;
    }

    label {
      display: block;
      font-size: 0.9rem;
      font-weight: 500;
      color: var(--text);
      margin-bottom: 8px;
    }

    .label-hint {
      color: var(--text-muted);
      font-size: 0.8rem;
      font-weight: 400;
      margin-left: 6px;
    }

    textarea, input[type="text"], select {
      width: 100%;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 14px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      transition: border-color 0.2s, box-shadow 0.2s;
    }

    textarea:focus, input[type="text"]:focus, select:focus {
      outline: none;
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .slider-container {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    input[type="range"] {
      flex: 1;
      accent-color: var(--accent-primary);
    }

    .btn-submit {
      width: 100%;
      padding: 14px;
      background: var(--accent-gradient);
      border: none;
      border-radius: 10px;
      color: white;
      font-family: inherit;
      font-size: 1.05rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: opacity 0.2s, transform 0.1s;
      box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    }

    .btn-submit:hover { opacity: 0.95; }
    .btn-submit:active { transform: scale(0.99); }
    .btn-submit:disabled { opacity: 0.5; cursor: not-allowed; }

    .result-section {
      margin-top: 24px;
      padding: 18px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      display: none;
    }

    audio {
      width: 100%;
      margin-top: 10px;
      border-radius: 8px;
    }

    .meta-bar {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 8px;
    }

    .code-box {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      overflow-x: auto;
      color: #cbd5e1;
      margin-top: 8px;
    }

    .endpoint-tag {
      background: #1e293b;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      color: #38bdf8;
    }

    .loader {
      border: 3px solid rgba(255,255,255,0.2);
      border-top: 3px solid #fff;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }

    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    footer {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: auto;
      padding-top: 20px;
    }
    footer a { color: #818cf8; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">
        <span class="status-dot"></span>
        <span id="server-status">API Server Ready • 48kHz Output</span>
      </div>
      <h1>VoxCPM Local API</h1>
      <p class="subtitle">Tokenizer-Free TTS • Multilingual • Voice Design • Controllable Cloning</p>
    </header>

    <div class="card">
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('design')">🎨 Voice Design</button>
        <button class="tab-btn" onclick="switchTab('clone')">🎛️ Voice Cloning</button>
        <button class="tab-btn" onclick="switchTab('openai')">🤖 OpenAI Compatible</button>
        <button class="tab-btn" onclick="switchTab('api-docs')">📖 API Docs</button>
      </div>

      <!-- TAB 1: Voice Design -->
      <div id="tab-design">
        <div class="form-group">
          <label>Target Text <span class="label-hint">(supports 30+ languages & dialects)</span></label>
          <textarea id="design-text" rows="3" placeholder="Enter text to synthesize...">VoxCPM2 brings studio-quality multilingual speech synthesis with true-to-life voice cloning.</textarea>
        </div>

        <div class="form-group">
          <label>Voice Description / Preset</label>
          <div class="grid-2" style="margin-bottom: 8px;">
            <select id="preset-selector" onchange="applyPreset()">
              <option value="">-- Choose a Preset Voice or describe below --</option>
              <option value="alloy">Alloy (Balanced, clear neutral)</option>
              <option value="echo">Echo (Warm, deep male)</option>
              <option value="nova">Nova (Bright, energetic young female)</option>
              <option value="shimmer">Shimmer (Gentle, soft melodic female)</option>
              <option value="storyteller">Storyteller (Documentary narrator)</option>
              <option value="coach">Coach (Energetic, motivational)</option>
              <option value="whisper">Whisper (Soft, breathy whispering)</option>
            </select>
            <input type="text" id="design-prompt" placeholder="Or custom description: (Young woman, gentle and sweet voice)">
          </div>
        </div>

        <div class="grid-2">
          <div class="form-group">
            <label>CFG Scale: <span id="cfg-val">2.0</span></label>
            <div class="slider-container">
              <input type="range" id="design-cfg" min="1.0" max="5.0" step="0.1" value="2.0" oninput="document.getElementById('cfg-val').innerText=this.value">
            </div>
          </div>
          <div class="form-group">
            <label>Inference Steps: <span id="steps-val">10</span></label>
            <div class="slider-container">
              <input type="range" id="design-steps" min="4" max="25" step="1" value="10" oninput="document.getElementById('steps-val').innerText=this.value">
            </div>
          </div>
        </div>

        <button class="btn-submit" id="btn-generate" onclick="generateVoiceDesign()">
          <span>Generate Speech</span>
        </button>

        <div class="result-section" id="design-result">
          <label>Generated Audio Result</label>
          <audio id="audio-player" controls autoplay></audio>
          <div class="meta-bar">
            <span id="meta-duration">Duration: --</span>
            <span id="meta-time">Latency: --</span>
            <span id="meta-rtf">RTF: --</span>
          </div>
        </div>
      </div>

      <!-- TAB 2: Voice Cloning -->
      <div id="tab-clone" style="display: none;">
        <div class="form-group">
          <label>Target Text to Synthesize</label>
          <textarea id="clone-text" rows="3" placeholder="Text to say with the cloned voice...">This is a cloned voice generated locally using VoxCPM.</textarea>
        </div>

        <div class="form-group">
          <label>Upload Reference Audio Clip <span class="label-hint">(.wav, .mp3, .m4a - 5 to 15s recommended)</span></label>
          <input type="file" id="clone-file" accept="audio/*">
        </div>

        <div class="form-group">
          <label>Control Guidance <span class="label-hint">(optional style control while preserving timbre)</span></label>
          <input type="text" id="clone-control" placeholder="e.g. slightly faster, cheerful tone, smiling">
        </div>

        <div class="form-group">
          <label>Reference Transcript <span class="label-hint">(optional - activates Ultimate Nuance Continuation mode)</span></label>
          <input type="text" id="clone-prompt-text" placeholder="Exact words spoken in the reference audio clip">
        </div>

        <button class="btn-submit" id="btn-clone" onclick="generateVoiceClone()">
          <span>Clone Voice & Synthesize</span>
        </button>

        <div class="result-section" id="clone-result">
          <label>Cloned Audio Result</label>
          <audio id="clone-player" controls autoplay></audio>
        </div>
      </div>

      <!-- TAB 3: OpenAI Compatible -->
      <div id="tab-openai" style="display: none;">
        <p style="margin-bottom: 14px; color: var(--text-muted); font-size: 0.95rem;">
          Connect your favorite apps (<strong style="color: #fff;">OpenWebUI</strong>, <strong style="color: #fff;">LibreChat</strong>, <strong style="color: #fff;">SillyTavern</strong>, Python OpenAI SDK) directly to this local endpoint:
        </p>
        <div class="code-box">Endpoint: POST http://localhost:8000/v1/audio/speech</div>

        <div class="form-group" style="margin-top: 16px;">
          <label>Input Text</label>
          <textarea id="openai-text" rows="2">Testing OpenAI compatible text-to-speech from VoxCPM.</textarea>
        </div>

        <div class="grid-2">
          <div class="form-group">
            <label>Voice</label>
            <select id="openai-voice">
              <option value="alloy">alloy (neutral clear)</option>
              <option value="echo">echo (warm deep male)</option>
              <option value="fable">fable (expressive storyteller)</option>
              <option value="onyx">onyx (authoritative baritone)</option>
              <option value="nova">nova (bright cheerful female)</option>
              <option value="shimmer">shimmer (soft melodic female)</option>
            </select>
          </div>
          <div class="form-group">
            <label>Speed</label>
            <input type="text" id="openai-speed" value="1.0">
          </div>
        </div>

        <button class="btn-submit" onclick="testOpenAIEndpoint()">
          <span>Test OpenAI Endpoint</span>
        </button>

        <div class="result-section" id="openai-result">
          <label>Audio Stream Result</label>
          <audio id="openai-player" controls autoplay></audio>
        </div>
      </div>

      <!-- TAB 4: API Docs -->
      <div id="tab-api-docs" style="display: none;">
        <h3 style="margin-bottom: 8px;">Interactive API Documentation</h3>
        <p style="color: var(--text-muted); margin-bottom: 16px;">Full interactive Swagger documentation and schemas are available at <a href="/docs" target="_blank" style="color: #818cf8;">/docs</a> and <a href="/redoc" target="_blank" style="color: #818cf8;">/redoc</a>.</p>

        <h4 style="margin-top: 14px;">1. cURL Example (Voice Design)</h4>
        <div class="code-box">curl -X POST http://localhost:8000/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to VoxCPM!",
    "voice_design": "(A cheerful young female voice)",
    "cfg_value": 2.0,
    "inference_timesteps": 10
  }' --output output.wav</div>

        <h4 style="margin-top: 14px;">2. Python (OpenAI SDK Compatible)</h4>
        <div class="code-box">from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
response = client.audio.speech.create(
    model="voxcpm2",
    voice="nova",
    input="Hello from the official OpenAI Python library!"
)
response.stream_to_file("speech.wav")</div>

        <h4 style="margin-top: 14px;">3. Python (Direct Requests Voice Cloning)</h4>
        <div class="code-box">import requests

url = "http://localhost:8000/v1/tts/clone"
files = {"file": open("sample.wav", "rb")}
data = {
    "text": "This voice is synthesized from a short sample.",
    "control": "cheerful, slightly smiling"
}
res = requests.post(url, files=files, data=data)
with open("cloned.wav", "wb") as f:
    f.write(res.content)</div>
      </div>

    </div>

    <footer>
      Powered by <a href="https://github.com/OpenBMB/VoxCPM" target="_blank">OpenBMB / VoxCPM</a> • Open-Source Tokenizer-Free Multilingual TTS
    </footer>
  </div>

  <script>
    function switchTab(tab) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      ['design', 'clone', 'openai', 'api-docs'].forEach(t => {
        document.getElementById('tab-' + t).style.display = (t === tab) ? 'block' : 'none';
      });
    }

    function applyPreset() {
      const p = document.getElementById('preset-selector').value;
      const presets = {
        'alloy': '(A balanced, versatile neutral voice with clear articulation)',
        'echo': '(A warm, deep, resonant male voice speaking in a calm, confident tone)',
        'nova': '(A bright, cheerful, energetic young female voice)',
        'shimmer': '(A gentle, soft, melodic and sweet young female voice)',
        'storyteller': '(An engaging documentary narrator, mature, rich tone)',
        'coach': '(High energy, motivational, fast-paced coach voice)',
        'whisper': '(A gentle, breathy, intimate whispering female voice)'
      };
      if (presets[p]) {
        document.getElementById('design-prompt').value = presets[p];
      }
    }

    async function generateVoiceDesign() {
      const btn = document.getElementById('btn-generate');
      const text = document.getElementById('design-text').value;
      const voice_design = document.getElementById('design-prompt').value;
      const cfg_value = parseFloat(document.getElementById('design-cfg').value);
      const inference_timesteps = parseInt(document.getElementById('design-steps').value);

      btn.disabled = true;
      btn.innerHTML = '<span class="loader"></span> Generating...';

      const t0 = performance.now();
      try {
        const res = await fetch('/v1/tts/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            text,
            voice_design: voice_design || null,
            cfg_value,
            inference_timesteps,
            response_format: 'wav'
          })
        });

        if (!res.ok) {
          const err = await res.json();
          alert('Error: ' + (err.detail || res.statusText));
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const player = document.getElementById('audio-player');
        player.src = url;

        const dur = res.headers.get('X-Audio-Duration') || '--';
        const genTime = res.headers.get('X-Generation-Time') || ((performance.now() - t0)/1000).toFixed(2);
        
        document.getElementById('meta-duration').innerText = 'Audio: ' + dur + 's';
        document.getElementById('meta-time').innerText = 'Generation: ' + genTime + 's';
        if (dur !== '--') {
          document.getElementById('meta-rtf').innerText = 'RTF: ' + (parseFloat(genTime) / parseFloat(dur)).toFixed(2);
        }
        document.getElementById('design-result').style.display = 'block';
      } catch (err) {
        alert('Network error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>Generate Speech</span>';
      }
    }

    async function generateVoiceClone() {
      const btn = document.getElementById('btn-clone');
      const fileInput = document.getElementById('clone-file');
      const text = document.getElementById('clone-text').value;
      const control = document.getElementById('clone-control').value;
      const prompt_text = document.getElementById('clone-prompt-text').value;

      if (!fileInput.files.length) {
        alert('Please select an audio file for reference cloning.');
        return;
      }

      btn.disabled = true;
      btn.innerHTML = '<span class="loader"></span> Cloning Voice...';

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('text', text);
      if (control) formData.append('control', control);
      if (prompt_text) formData.append('prompt_text', prompt_text);

      try {
        const res = await fetch('/v1/tts/clone', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const err = await res.json();
          alert('Error: ' + (err.detail || res.statusText));
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const player = document.getElementById('clone-player');
        player.src = url;
        document.getElementById('clone-result').style.display = 'block';
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>Clone Voice & Synthesize</span>';
      }
    }

    async function testOpenAIEndpoint() {
      const text = document.getElementById('openai-text').value;
      const voice = document.getElementById('openai-voice').value;
      const speed = parseFloat(document.getElementById('openai-speed').value) || 1.0;

      try {
        const res = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            model: 'voxcpm2',
            input: text,
            voice,
            speed,
            response_format: 'wav'
          })
        });

        if (!res.ok) {
          const err = await res.json();
          alert('Error: ' + (err.detail || res.statusText));
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const player = document.getElementById('openai-player');
        player.src = url;
        document.getElementById('openai-result').style.display = 'block';
      } catch (err) {
        alert('Error: ' + err.message);
      }
    }

    // Health check on load
    fetch('/health').then(r => r.json()).then(data => {
      document.getElementById('server-status').innerText = 
        `API Online • ${data.device.toUpperCase()} • ${data.sample_rate / 1000}kHz Output`;
    }).catch(() => {});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

def main():
    parser = argparse.ArgumentParser(description="VoxCPM Local API Server")
    parser.add_argument("--host", type=str, default=os.getenv("VOXCPM_HOST", "0.0.0.0"), help="Host interface (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("VOXCPM_PORT", "8000")), help="Port to listen on (default 8000)")
    parser.add_argument("--model-id", type=str, default=os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2"), help="Model ID or local directory")
    parser.add_argument("--device", type=str, default=os.getenv("VOXCPM_DEVICE", "auto"), help="Device: auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--load-denoiser", action="store_true", default=os.getenv("VOXCPM_LOAD_DENOISER", "false").lower() in ("true", "1", "yes"), help="Load speech ZipEnhancer denoiser")
    parser.add_argument("--optimize", action="store_true", default=os.getenv("VOXCPM_OPTIMIZE", "false").lower() in ("true", "1", "yes"), help="Enable torch.compile optimization")
    parser.add_argument("--preload", action="store_true", default=os.getenv("VOXCPM_PRELOAD", "false").lower() in ("true", "1", "yes"), help="Preload model at server launch rather than lazy loading")
    args = parser.parse_args()

    state.model_id = args.model_id
    state.device = args.device
    state.load_denoiser = args.load_denoiser
    state.optimize = args.optimize

    print("=" * 60)
    print(" 🚀 VoxCPM Local API Server")
    print(f" • Host & Port: http://{args.host}:{args.port}")
    print(f" • Model:       {args.model_id}")
    print(f" • Device:      {args.device}")
    print(f" • Web UI:      http://localhost:{args.port}/ui")
    print(f" • API Docs:    http://localhost:{args.port}/docs")
    print(f" • OpenAI Spec: http://localhost:{args.port}/v1/audio/speech")
    print("=" * 60)

    if args.preload:
        logger.info("Preloading model in background at startup...")

        def _preload():
            try:
                get_or_load_model()
            except Exception:
                pass  # already logged in get_or_load_model

        threading.Thread(target=_preload, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

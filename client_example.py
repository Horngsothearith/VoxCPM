"""
VoxCPM API Client Examples
==========================
Shows how to interact with the local VoxCPM API server using:
1. Standard `requests` library for direct VoxCPM features (Voice Design & Cloning)
2. Official `openai` Python SDK for drop-in OpenAI TTS compatibility
"""

import sys
import requests

SERVER_URL = "http://localhost:8000"

def test_health():
    print("\n--- 1. Checking Health & Server Status ---")
    try:
        res = requests.get(f"{SERVER_URL}/health")
        print(f"Status Code: {res.status_code}")
        print("Response:", res.json())
    except requests.exceptions.ConnectionError:
        print("⚠️ Could not connect to API server. Ensure `python api_server.py` is running.")
        return False
    return True

def test_voice_design():
    print("\n--- 2. Generating Speech with Voice Design ---")
    payload = {
        "text": "Hello! I am a speech model running completely locally on your machine.",
        "voice_design": "(A cheerful young female voice, clear and warm)",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "response_format": "wav"
    }
    res = requests.post(f"{SERVER_URL}/v1/tts/generate", json=payload)
    if res.status_code == 200:
        filename = "test_voice_design.wav"
        with open(filename, "wb") as f:
            f.write(res.content)
        print(f"Saved generated audio to: {filename}")
        print(f"Headers: Audio-Duration={res.headers.get('X-Audio-Duration')}s, Gen-Time={res.headers.get('X-Generation-Time')}s")
    else:
        print(f"Error {res.status_code}:", res.text)

def test_openai_compatibility():
    print("\n--- 3. Testing OpenAI-Compatible /v1/audio/speech Endpoint ---")
    payload = {
        "model": "voxcpm2",
        "input": "This is synthesized using the OpenAI-compatible speech endpoint.",
        "voice": "nova",
        "response_format": "wav"
    }
    res = requests.post(f"{SERVER_URL}/v1/audio/speech", json=payload)
    if res.status_code == 200:
        filename = "test_openai_tts.wav"
        with open(filename, "wb") as f:
            f.write(res.content)
        print(f"Saved OpenAI-compatible TTS audio to: {filename}")
    else:
        print(f"Error {res.status_code}:", res.text)

def test_voice_cloning(sample_wav_path: str = "examples/reference_speaker.wav"):
    print(f"\n--- 4. Testing Voice Cloning with File Upload ({sample_wav_path}) ---")
    import os
    if not os.path.exists(sample_wav_path):
        print(f"Sample audio not found at {sample_wav_path}. Skipping clone test.")
        return

    with open(sample_wav_path, "rb") as f:
        files = {"file": (os.path.basename(sample_wav_path), f, "audio/wav")}
        data = {
            "text": "This voice has been cloned from the reference sample audio.",
            "control": "warm and friendly",
            "cfg_value": 2.0,
            "inference_timesteps": 10
        }
        res = requests.post(f"{SERVER_URL}/v1/tts/clone", files=files, data=data)
        if res.status_code == 200:
            filename = "test_cloned_voice.wav"
            with open(filename, "wb") as out_f:
                out_f.write(res.content)
            print(f"Saved cloned voice audio to: {filename}")
        else:
            print(f"Error {res.status_code}:", res.text)

if __name__ == "__main__":
    if test_health():
        test_voice_design()
        test_openai_compatibility()
        test_voice_cloning()

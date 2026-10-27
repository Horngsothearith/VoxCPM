"""Regression guards for the API input and CORS security boundaries."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = REPO_ROOT / "api_server.py"


def _api_tree() -> ast.Module:
    return ast.parse(API_SOURCE.read_text(encoding="utf-8"), filename=str(API_SOURCE))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _field_max_length(class_node: ast.ClassDef, field_name: str) -> ast.expr | None:
    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != field_name or not isinstance(node.value, ast.Call):
            continue
        for keyword in node.value.keywords:
            if keyword.arg == "max_length":
                return keyword.value
    return None


def _function_source(tree: ast.Module, name: str) -> str:
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
    return ast.get_source_segment(API_SOURCE.read_text(encoding="utf-8"), function) or ""


def test_speech_request_fields_have_bounded_lengths():
    tree = _api_tree()
    speech = _find_class(tree, "OpenAISpeechRequest")
    generate = _find_class(tree, "TTSGenerateRequest")

    for field_name in ("input", "voice"):
        assert isinstance(_field_max_length(speech, field_name), ast.Name)
        assert _field_max_length(speech, field_name).id == "MAX_TEXT_LENGTH"

    for field_name in ("text", "voice_design", "prompt_text"):
        assert isinstance(_field_max_length(generate, field_name), ast.Name)
        assert _field_max_length(generate, field_name).id == "MAX_TEXT_LENGTH"

    for field_name in ("reference_audio_base64", "prompt_audio_base64"):
        assert isinstance(_field_max_length(generate, field_name), ast.Name)
        assert _field_max_length(generate, field_name).id == "MAX_AUDIO_BASE64_LENGTH"


def test_cors_wildcard_does_not_allow_credentials():
    tree = _api_tree()
    middleware_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_middleware"
    )
    credentials = next(keyword for keyword in middleware_call.keywords if keyword.arg == "allow_credentials")
    assert isinstance(credentials.value, ast.Constant)
    assert credentials.value.value is False


def test_clone_endpoint_streams_and_bounds_uploads():
    source = API_SOURCE.read_text(encoding="utf-8")
    function_source = _function_source(_api_tree(), "tts_clone_file")

    assert "MAX_UPLOAD_SIZE = 25 * 1024 * 1024" in source
    assert "ALLOWED_AUDIO_EXTENSIONS" in source
    assert "await file.read(chunk_size)" in function_source
    assert "total_bytes += len(chunk)" in function_source
    assert "total_bytes > MAX_UPLOAD_SIZE" in function_source
    assert "status_code=413" in function_source
    assert "suffix not in ALLOWED_AUDIO_EXTENSIONS" in function_source
    assert "except HTTPException" in function_source
    assert "if temp_ref_path and os.path.exists(temp_ref_path)" in function_source


def test_rejected_clone_uploads_are_not_charged(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.syspath_prepend(str(REPO_ROOT))
    import accounts

    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.db")  # api_server runs init_db() on import
    import api_server

    charged = []
    monkeypatch.setattr(accounts, "AUTH_ENABLED", False)
    monkeypatch.setattr(accounts, "charge", lambda *args: charged.append(args) or 0)
    monkeypatch.setattr(api_server, "get_or_load_model", lambda: object())
    monkeypatch.setattr(api_server, "MAX_UPLOAD_SIZE", 8)
    client = TestClient(api_server.app)

    def clone(filename, content):
        return client.post("/v1/tts/clone", data={"text": "hi"}, files={"file": (filename, content)}).status_code

    assert clone("ref.wav", b"x" * 9) == 413
    assert clone("ref.wav", b"") == 400
    assert clone("ref.exe", b"x") == 400
    assert charged == []

"""TBS: the OpenAI-compatible ASR adapter keeps per-segment word timing when the server sends it."""
from openai.types.audio import TranscriptionVerbose

from services.asr_backend import OpenAICompatASRBackend


def _seg(text, start, end, **extra):
    return {"id": 0, "seek": 0, "start": start, "end": end, "text": text, "tokens": [], "temperature": 0,
            "avg_logprob": 0, "compression_ratio": 0, "no_speech_prob": 0, **extra}


def test_words_inside_segments_become_word_chunks():
    resp = TranscriptionVerbose.model_validate({
        "task": "transcribe", "language": "bengali", "duration": 4.0, "text": "আমরা আসলে",
        "segments": [_seg("আমরা আসলে", 0.1, 2.0, words=[
            {"word": "আমরা", "start": 0.1, "end": 1.0}, {"word": "আসলে", "start": 1.0, "end": 2.0}])],
    })
    out = OpenAICompatASRBackend._adapt_response(resp)
    assert out["segments"][0]["words"] == [{"text": "আমরা", "start": 0.1, "end": 1.0},
                                           {"text": "আসলে", "start": 1.0, "end": 2.0}]
    assert out["chunks"] == [{"text": "আমরা", "timestamp": (0.1, 1.0)}, {"text": "আসলে", "timestamp": (1.0, 2.0)}]
    assert out["language"] == "bengali"


def test_plain_openai_segments_unchanged():
    resp = TranscriptionVerbose.model_validate({
        "task": "transcribe", "language": "english", "duration": 2.0, "text": "hello there",
        "segments": [_seg(" hello there", 0.0, 2.0)],
    })
    out = OpenAICompatASRBackend._adapt_response(resp)
    assert out["segments"][0]["words"] == []
    assert out["chunks"] == [{"text": "hello there", "timestamp": (0.0, 2.0)}]

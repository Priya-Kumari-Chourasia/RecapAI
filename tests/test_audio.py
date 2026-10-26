from types import SimpleNamespace

from pydub.generators import Sine

from core import transcriber
from utils import audio_processor


def test_long_audio_is_chunked_with_correct_offsets(tmp_path):
    tone = Sine(440).to_audio_segment(duration=25_000)  # 25 s
    path = tmp_path / "talk.wav"
    tone.export(path, format="wav")

    audio = audio_processor.load_audio(str(path))
    assert audio.channels == 1 and audio.frame_rate == 16000

    # 10 s chunks -> 3 chunks starting at 0, 10, 20 seconds
    chunks = audio_processor.chunk_audio(audio, str(tmp_path / "talk"), chunk_minutes=10 / 60)
    assert [round(offset) for _, offset in chunks] == [0, 10, 20]
    assert all(p.endswith(".mp3") for p, _ in chunks)


def test_groq_verbose_json_becomes_shifted_segments(monkeypatch, tmp_path):
    fake_response = SimpleNamespace(model_dump=lambda: {
        "text": "hello world",
        "segments": [{"start": 0.0, "end": 2.0, "text": " hello"}, {"start": 2.0, "end": 3.5, "text": " world"}],
    })
    calls = []

    class FakeTranscriptions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return fake_response

    monkeypatch.setattr(transcriber, "_groq", lambda: SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions())))
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")

    segments = transcriber.transcribe_all([(str(chunk), 0.0), (str(chunk), 600.0)], engine="groq")
    assert calls[0]["response_format"] == "verbose_json"
    assert [s["start"] for s in segments] == [0.0, 2.0, 600.0, 602.0]
    assert segments[-1] == {"start": 602.0, "end": 603.5, "text": "world"}


def test_vocabulary_is_sent_to_whisper_as_prompt(monkeypatch, tmp_path):
    calls = []

    class FakeTranscriptions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {"segments": [{"start": 0, "end": 1, "text": "pre-training"}]}

    monkeypatch.setattr(transcriber, "_groq", lambda: SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions())))
    chunk = tmp_path / "c.mp3"
    chunk.write_bytes(b"fake")
    transcriber.transcribe_all([(str(chunk), 0.0)], engine="groq", vocabulary="Bonsai,  XMC\npre-training, ")
    assert calls[0]["prompt"] == "Glossary: Bonsai, XMC, pre-training."
    transcriber.transcribe_all([(str(chunk), 0.0)], engine="groq", vocabulary="  ")
    assert "prompt" not in calls[1]

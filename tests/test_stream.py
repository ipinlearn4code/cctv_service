import pytest
from app.services.stream_processor import generate_frames

def test_stream_processor():
    try:
        frames = generate_frames("invalid_id")
        next(frames)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True
# wan/utils/duration.py

def seconds_to_frames(seconds: int, fps: int = 16) -> int:
    if seconds <= 0:
        raise ValueError("Duration must be > 0 seconds")
    return seconds * fps

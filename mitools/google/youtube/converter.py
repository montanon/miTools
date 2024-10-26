from os import PathLike
from pathlib import Path

from moviepy.editor import VideoFileClip


def video_to_audio(video_path: PathLike, audio_path: Path):
    try:
        video_path, audio_path = Path(video_path), Path(audio_path)
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(audio_path))
        clip.close()
        print(f"Audio saved to {audio_path}")
    except Exception as e:
        print(f"Error converting video to audio: {str(e)}")

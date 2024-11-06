from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import List

from pytubefix import YouTube
from tqdm import tqdm

from mitools.exceptions import ArgumentValueError


def download_video(
    url: str, output_path: Path, resolution: str = "720p", recalculate: bool = False
):
    if output_path.exists() and not recalculate:
        raise ArgumentValueError("Output path already exists.")
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(res=resolution, file_extension="mp4").first()
        if not stream:
            raise ArgumentValueError(f"Resolution {resolution} not available.")
        print(f"Downloading: {yt.title}")
        stream.download(output_path)
        print("Download completed successfully!")
    except Exception as e:
        raise RuntimeError(f"Error downloading video: {str(e)}")


def download_audio_video(url: str, output_path: PathLike):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise ArgumentValueError("Couldn't get audio stream.")
        print(f"Downloading: {yt.title}")
        stream.download(output_path)
        print("Download completed successfully!")
    except Exception as e:
        raise RuntimeError(f"Error downloading audio: {str(e)}")


def batch_download(urls: List[str], output_path: Path, resolution: str = "720p"):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download_video, url, output_path, resolution)
            for url in urls
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during batch download: {str(e)}")

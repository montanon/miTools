import json
from pathlib import Path

from pytubefix import YouTube


def extract_metadata(url: str) -> dict:
    yt = YouTube(url)
    metadata = {
        "title": yt.title,
        "description": yt.description,
        "length": yt.length,
        "views": yt.views,
        "author": yt.author,
        "channel_id": yt.channel_id,
        "channel_url": yt.channel_url,
        "keywords": yt.keywords,
        "metadata": yt.metadata,
        "publish_date": yt.publish_date,
        "rating": yt.rating,
    }
    return metadata


def save_metadata(url: str, output_file: Path):
    metadata = extract_metadata(url)
    with output_file.open("w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {output_file}")

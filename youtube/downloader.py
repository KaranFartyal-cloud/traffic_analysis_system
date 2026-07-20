"""
downloader.py

Responsible for:
1. Validating YouTube URLs
2. Extracting the highest quality stream
3. Returning the direct stream URL
"""

from yt_dlp import YoutubeDL


class YouTubeDownloader:

    def __init__(self):
        self.options = {
            "quiet": True,
            "noplaylist": True,
            "format": "best"
        }

    def get_stream_url(self, youtube_url):
        """
        Returns the direct video stream URL.

        Parameters
        ----------
        youtube_url : str

        Returns
        -------
        dict
            {
                "title": "...",
                "stream_url": "...",
                "fps": 30,
                "width": 1920,
                "height": 1080
            }
        """

        try:

            with YoutubeDL(self.options) as ydl:

                info = ydl.extract_info(
                    youtube_url,
                    download=False
                )

            return {
                "title": info.get("title"),
                "stream_url": info.get("url"),
                "fps": info.get("fps"),
                "width": info.get("width"),
                "height": info.get("height")
            }

        except Exception as e:
            raise Exception(f"Unable to extract stream.\n{e}")
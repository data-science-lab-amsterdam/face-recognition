import requests
import cv2
import numpy as np
from basecamera import BaseCamera
import logging


class NetworkCamera(BaseCamera):
    url = 'http://166.155.203.48/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000'

    @staticmethod
    def set_url(url):
        NetworkCamera.url = url

    @staticmethod
    def frames():
        # define stream and first read
        try:
            connection = requests.get(NetworkCamera.url, stream=True)
            connection.raise_for_status()
        except Exception as e:
            logging.error("Could not connect to network camera feed")
            raise e

        bytestream = bytes()
        streamer = connection.iter_content(chunk_size=10 * 1024)

        for chunk in streamer:
            # add chunk to bytestream
            bytestream += chunk

            # find start & end bytes defining jpg image
            jpg_start = bytestream.find(b'\xff\xd8')
            jpg_end = bytestream.find(b'\xff\xd9')
            if jpg_start != -1 and jpg_end != -1:
                # jpg image found: set it as the current frame
                jpg = bytestream[jpg_start:jpg_end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                yield frame

                # truncate bytestream: remove everything before last image
                bytestream = bytestream[jpg_end + 2:]

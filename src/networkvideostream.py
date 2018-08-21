import cv2
import logging
import threading
import requests
import numpy as np
from videostream import BaseVideoStream
from utils import FPSCounter


class NetworkVideoStream(BaseVideoStream):
    """
    VideoStream implementation for the webcam
    """

    def __init__(self, url, display=False, count_fps=False):
        # parent constructor
        super().__init__(display, count_fps)

        # define stream and first read
        try:
            connection = requests.get(url, stream=True)
            connection.raise_for_status()
        except Exception as e:
            logging.error("Could not connect to network camera feed")
            raise e
        self.bytestream = bytes()
        self.streamer = connection.iter_content(chunk_size=10*1024)
        self._get_first_frame()

    def _get_first_frame(self):
        """
        Ugly solution, but just to get the first frame
        """
        for chunk in self.streamer:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return  # this also ends the thread

            # add chunk to bytestream
            self.bytestream += chunk

            # find start & end bytes defining jpg image
            jpg_start = self.bytestream.find(b'\xff\xd8')
            jpg_end = self.bytestream.find(b'\xff\xd9')
            if jpg_start != -1 and jpg_end != -1:
                # jpg image found: set it as the current frame
                jpg = self.bytestream[jpg_start:jpg_end + 2]
                (self.grabbed, self.frame) = (True, cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR))
                return

    def _update(self):
        """
        keep looping infinitely until the thread is stopped
        """
        for chunk in self.streamer:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return  # this also ends the thread

            # add chunk to bytestream
            self.bytestream += chunk

            # find start & end bytes defining jpg image
            jpg_start = self.bytestream.find(b'\xff\xd8')
            jpg_end = self.bytestream.find(b'\xff\xd9')
            if jpg_start != -1 and jpg_end != -1:
                # jpg image found: set it as the current frame
                jpg = self.bytestream[jpg_start:jpg_end + 2]
                (self.grabbed, self.frame) = (True, cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR))

                # truncate bytestream: remove everything before last image
                self.bytestream = self.bytestream[jpg_end + 2:]

                if self.display:
                    # Display the resulting image
                    cv2.imshow('Video', self.frame)

                    # Hit 'q' on the keyboard to quit!
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop()

                if self.count_fps:
                    self.fps.update()

    def stop(self):
        """
        indicate that the thread should be stopped
        """
        logging.info("Network cam video stream stopped")
        self.stopped = True

        if self.count_fps:
            self.fps.stop()

        if self.display:
            cv2.destroyAllWindows()




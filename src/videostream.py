from abc import ABC, abstractmethod
import cv2
import threading
import numpy as np
from utils import FPSCounter
import logging


class BaseVideoStream(ABC):
    """
    Abstract base class
    """
    def __init__(self, display=False, count_fps=False):
        """
        initialize the video camera stream and read the first frame from the stream
        """
        self.display = display

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

        self.count_fps = count_fps
        if self.count_fps:
            self.fps = FPSCounter()

    def start(self):
        """
        start the thread to read frames from the video stream
        """
        self.stopped = False
        threading.Thread(target=self._update, args=()).start()
        if self.count_fps:
            self.fps.start()

        return self

    @abstractmethod
    def _update(self):
        """
        should be implemented by child class
        """
        pass

    def read(self):
        """
        return the frame most recently read
        """
        if self.stopped:
            return False, False

        return self.grabbed, self.frame

    def stop(self):
        """
        indicate that the thread should be stopped
        """
        logging.info("Webcam video stream stopped")
        self.stopped = True

        if self.count_fps:
            self.fps.stop()

        if self.display:
            cv2.destroyAllWindows()

    def get_fps(self):
        """
        Return elapsed fps of fps counter
        """
        if not self.count_fps:
            logging.error("No FPSCounter set")
            return None
        return self.fps.get_fps()

    @staticmethod
    def to_jpeg_bytes(thing):
        if isinstance(thing, str):
            # filename: just read the image contents
            with open(thing, 'rb') as f:
                return f.read()
        elif isinstance(thing, np.ndarray):
            # pixel array: encode to jpeg
            ok, jpg = cv2.imencode('.jpg', thing)
            if not ok:
                raise ValueError('Could not encode object to jpg')
            return jpg.tobytes()
        else:
            raise ValueError("Cannot encode object of type {} to jpeg".format(type(thing)))

    def __del__(self):
        self.stop()



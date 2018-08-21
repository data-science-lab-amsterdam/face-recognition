from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2
import logging
from utils import FPSCounter
from videostream import BaseVideoStream


class PicamVideoStream(BaseVideoStream):
    """
    VideoStream implementation for the PiCamera
    """
    def __init__(self, display=False, count_fps=True, resolution=(1280, 720), framerate=20):
        # call parent constructor
        super().__init__(display, count_fps)

        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.raw_capture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True)

        # initial frame
        self.frame = next(self.stream).array
        self.raw_capture.truncate(0)
        self.grabbed = True

    def _update(self):
        """
        Read from the camera until the thread is stopped
        """
        for f in self.stream:
            try:
                # grab the frame from the stream and clear the stream in preparation for the next frame
                self.frame = f.array
                self.grabbed = True
                self.raw_capture.truncate(0)
            except Exception as e:
                logging.error("Something went wrong while reading from the camera")
                logging.error(e)
                self.grabbed, self.frame = (False, False)

            # this stops the thread
            if self.stopped:
                return

            # update fps counter
            if self.count_fps:
                self.fps.update()

    def stop(self):
        """
        Custom actions when stopping
        """
        self.stopped = True

        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

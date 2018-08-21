import cv2
import logging
import threading
from videostream import BaseVideoStream
from utils import FPSCounter


class WebcamVideoStream(BaseVideoStream):
    """
    VideoStream implementation for the webcam
    """

    def __init__(self, device_id=0, display=False, count_fps=False):
        # parent constructor
        super().__init__(display, count_fps)

        # define stream and first read
        self.stream = cv2.VideoCapture(device_id)
        (self.grabbed, self.frame) = self.stream.read()

    def _update(self):
        """
        keep looping infinitely until the thread is stopped
        """
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return  # this also ends the thread

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

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
        logging.info("Webcam video stream stopped")
        self.stopped = True
        self.stream.release()

        if self.count_fps:
            self.fps.stop()

        if self.display:
            cv2.destroyAllWindows()




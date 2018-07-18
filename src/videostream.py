import cv2
import threading
from utils import FPSCounter
import logging


class WebcamVideoStream:
    def __init__(self, device_id=0, display=False, count_fps=False, decorator_func=None):
        """
        initialize the video camera stream and read the first frame from the stream
        """
        self.display = display
        self.stream = cv2.VideoCapture(device_id)
        (self.grabbed, self.frame) = self.stream.read()
        self.decorator_func = decorator_func

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

        self.count_fps = count_fps
        if self.count_fps:
            self.fps = FPSCounter()

    def start(self):
        """
        start the thread to read frames from the video stream
        """
        threading.Thread(target=self._update, args=()).start()
        if self.count_fps:
            self.fps.start()

        return self

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
        self.stream.release()

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

    def __del__(self):
        self.stop()



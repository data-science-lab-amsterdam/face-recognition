import cv2
from basecamera import BaseCamera


class OpencvCamera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        OpencvCamera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(OpencvCamera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # # encode as a jpeg image and return it
            # yield cv2.imencode('.jpg', img)[1].tobytes()
            yield img

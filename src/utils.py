from datetime import datetime


class FPSCounter:
    def __init__(self):
        """
        store the start time, end time, and total number of frames that were examined between start and end
        """
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        """
        start the timer
        """
        self._start = datetime.now()
        return self

    def stop(self):
        """
        reset the timer
        """
        self._start = datetime.now()
        self._num_frames = 0

    def update(self):
        """
        increment the total number of frames examined during the start and end intervals
        """
        self._num_frames += 1

    def get_fps(self):
        """
        compute the (approximate) frames per second
        """
        return self._num_frames / (datetime.now() - self._start).total_seconds()

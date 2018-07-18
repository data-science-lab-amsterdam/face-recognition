import logging
import argparse
from detector import DetectionApp


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] (%(threadName)-9s) %(message)s')

# parse arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-d', '--display', action='store_true', default=False,	help='Use to display video frames')
# args = ap.parse_args()
#
# FLG_DISPLAY = args.display

config = {
    'display': True,
    'speak': False,
    'camera_device_id': 0,
    'faces': {
        'detect': True,
        'shrink_frames': True,
        'anchor_images_path': './images'
    },
    'objects': {
        'detect': False,
        'shrink_frames': False
    }
}

app = DetectionApp(config=config)
app.start()

app.stop()

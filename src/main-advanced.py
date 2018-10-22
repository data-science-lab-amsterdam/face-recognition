#########################################################
#
# 'Advanced' and multi-threaded version
#
#########################################################

import logging
import argparse
from detector import DetectionApp


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] (%(threadName)-9s) %(message)s')

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', action='store_true', default=False,	help='Use to display video frames')
ap.add_argument('-s', '--sound', action='store_true', default=False,	help='Use to speak about recognized results')
ap.add_argument('-n', '--network-mode', dest='network_mode', action='store_true', default=False, help='Use to use a network video feed instead of built-in camera')
args = ap.parse_args()

logging.info('Running with display mode {}'.format('on' if args.display else 'off'))
logging.info('Running with sound mode {}'.format('on' if args.sound else 'off'))
logging.info('Running with network mode {}'.format('on' if args.network_mode else 'off'))

config = {
    'display': args.display,
    'speak': args.sound,
    'camera_device_id': ('network' if args.network_mode else 0),
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

with DetectionApp(config=config) as app:
    app.start()



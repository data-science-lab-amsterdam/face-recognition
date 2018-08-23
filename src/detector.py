import face_recognition
from pathlib import Path
from datetime import datetime, timedelta
import cv2
import os
import logging
import threading
import subprocess
import numpy as np
from collections import deque
import tensorflow as tf
from utils import FPSCounter
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def override_dict_values(d1, d2):
    """
    Override dictionary only for keys where a value is specified
    """
    new = d1.copy()
    for k, v in d2.items():
        if isinstance(v, dict):
            new[k] = override_dict_values(new[k], d2[k])
        else:
            new[k] = v

    return new


class DetectionApp:

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
            'shrink_frames': True
        }
    }

    def __init__(self, config):
        self.config = override_dict_values(self.config, config)

        # shortcut settings
        self.faces = self.config['faces']['detect']
        self.objects = self.config['objects']['detect']

        # use the appropriate videostreamer depending on the platform/camera to use
        if self.config['camera_device_id'] == 'pi':
            # only import if needed because it requires specific packages!
            from picamvideostream import PicamVideoStream

            self.video_stream = PicamVideoStream(display=False, count_fps=True)
        elif self.config['camera_device_id'] == 'network':
            # only import if needed because it requires specific packages!
            from networkvideostream import NetworkVideoStream

            self.video_stream = NetworkVideoStream(url='http://192.168.1.163:8554/stream/cam_pic_new.php',
                                                   display=False,
                                                   count_fps=True)
        else:
            # only import if needed because it requires specific packages!
            from webcamvideostream import WebcamVideoStream

            self.video_stream = WebcamVideoStream(device_id=self.config['camera_device_id'],
                                                  display=False,
                                                  count_fps=True)

        # set face recorgnition
        if self.faces:
            self.face_recognizer = FaceRecognizer(self.config['faces']['anchor_images_path'], count_fps=True)
            self._seen_faces = deque(maxlen=20)

        # set object recognition
        if self.objects:
            self.object_recognizer = ObjectRecognizer(count_fps=True)

        self.fps_counter = FPSCounter()

        # to prevent talking too much
        self.last_speech = datetime.now() - timedelta(minutes=10)

    def start(self):
        """
        Start the detection process:
        - video stream starts in a new thread
        - (optional) face recognition starts in a new thread
        - (optional) object recognition starts in a new thread
        - The main thread
          -
        """
        self.video_stream.start()
        if self.faces:
            self.face_recognizer.start(frame_reader=self.video_stream.read, shrink_frames=self.config['faces']['shrink_frames'])
        if self.objects:
            self.object_recognizer.start(frame_reader=self.video_stream.read)
        self.fps_counter.start()

        while True:
            # read a frame from the camera
            ok, frame = self.video_stream.read()
            if not ok:
                logging.info("Stopped!")
                break

            # get frame rates
            fps_data = {
                'webcam': self.video_stream.get_fps(),
                'main': self.fps_counter.get_fps()
            }

            # get face info from the face recognizer
            if self.faces:
                faces_data = self.face_recognizer.read()
                fps_data['face_detection'] = self.face_recognizer.get_fps()

            if self.objects:
                objects_data = self.object_recognizer.read()
                fps_data['object_detection'] = self.object_recognizer.get_fps()

            if self.config['display']:
                # show face information on the frame
                if self.faces:
                    DetectionApp.paint_faces_data(frame, faces_data)

                # show object information on the frame
                if self.objects:
                    DetectionApp.paint_objects_data(frame, objects_data)

                # show fps info on frame
                DetectionApp.paint_fps_data(frame, fps_data)

                # show the image
                cv2.imshow('Video', frame)
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.config['speak']:
                if self.faces:
                    self._talk_about(about='faces', data=faces_data)

                if self.objects:
                    self._talk_about(about='objects', data=objects_data)

            self.fps_counter.update()

            logging.info('Camera fps: {}'.format(fps_data['webcam']))
            if self.faces:
                logging.info('Face detection fps: {}'.format(fps_data['face_detection']))
            if self.objects:
                logging.info('Object detection fps: {}'.format(fps_data['object_detection']))
            logging.info('Main fps: {}'.format(fps_data['main']))

        self.stop()

    def stop(self):
        self.video_stream.stop()
        if self.faces:
            self.face_recognizer.stop()
        if self.objects:
            self.object_recognizer.stop()
        self.fps_counter.stop()

    def _talk_about(self, about, data):
        """
        Say some stuff about detected faces or objects
        """
        if not data:
            return

        if about == 'faces':
            for face in data:
                self._seen_faces.append(face['identity'])

            all_people = list(set(self._seen_faces))
            all_known_people = [name for name in all_people if name is not None]

            if len(all_known_people) == 0:
                text = "Hello. I haven't seen you before. Welcome!"
            elif len(all_known_people) == 1:
                text = "Hello {}".format(all_known_people[0])
            elif len(all_people) <= 3:
                all_but_last = ', '.join([name or 'stranger' for name in all_people[:-1]])
                last = all_people[-1] or 'stranger'
                names = ' and '.join([all_but_last, last])
                text = 'Hello {}'.format(names)
            elif len(all_known_people) > 3:
                text = 'Hi everybody. What a crowd! Why don\'t you get back to work?'
            self._say_prudent(text)

    def _say_prudent(self, text):
        """
        Say something unless you already just did
        """
        if datetime.now() - self.last_speech > timedelta(seconds=5):
            DetectionApp._say(text)
            self.last_speech = datetime.now()
            self._seen_faces.clear()

    @staticmethod
    def _say(text):
        """
        Actually make a sound
        Btw this implementation only works on a mac!
        """
        subprocess.call(['say', text])

    @staticmethod
    def paint_faces_data(frame, faces_data):
        """
        Paint boxes and labels/names around detected faces
        """
        for face in faces_data:
            (top, right, bottom, left) = face['location']

            if face['identity'] is None:
                name = 'Unknown'
                color = (0, 0, 255)  # red
            else:
                name = face['identity']
                color = (0, 128, 0)  # dark green

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    @staticmethod
    def paint_objects_data(frame, objects_data):
        """
        Paint boxes and labels for detected objects on given frame
        """
        try:
            (boxes, scores, classes, num_detections), category_index = objects_data
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )
        except ValueError:
            pass

    @staticmethod
    def paint_fps_data(frame, fps_data):
        """
        Show fps on frame
        """
        frame_height, frame_width, _ = frame.shape
        text = 'running @ {:.0f} fps'.format(fps_data['main'])
        cv2.putText(img=frame,
                    text=text,
                    org=(10, frame_height - 20),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=(0, 0, 255),
                    thickness=1
                    )

    def __enter__(self):
        return self

    def __exit__(self):
        self.stop()


############################## END CLASS ####################################


class FaceRecognizer:

    MAX_DISTANCE = 0.6

    def __init__(self, image_path, count_fps=False):
        self.image_path = image_path
        self.stopped = False
        self.database = {}
        self.known_face_names = None
        self.known_face_encodings = None
        self.face_data = []

        self.count_fps = count_fps
        if self.count_fps:
            self.fps = FPSCounter()

        self.setup_database()

    def setup_database(self):
        """
        Images a set of images in a certain directory as anchor images
        """
        for filename in Path(self.image_path).glob('*.jpg'):
            # load image
            image = face_recognition.load_image_file(filename)
            # use the name in the filename as the identity key
            identity = os.path.splitext(os.path.basename(str(filename)))[0].split('-')[0]
            # detect faces and get the model encoding of the first face
            self.database[identity] = face_recognition.face_encodings(image)[0]
            logging.info("Image of '{}' added to database".format(identity))

        self.known_face_names = list(self.database.keys())
        self.known_face_encodings = list(self.database.values())

    def read(self):
        return self.face_data

    def start(self, frame_reader, shrink_frames=False):
        self.stopped = False
        threading.Thread(target=self._update, args=(frame_reader, shrink_frames)).start()

        if self.count_fps:
            self.fps.start()

    def _update(self, frame_reader, shrink_frames=False):
        while True:
            if self.stopped:
                logging.info("Face detection stopped. Quitting thread...")
                return

            ok, frame = frame_reader()
            if not ok:
                logging.info("FaceRecognizer could not read frame. Stopping...")
                return

            if shrink_frames:
                # shrink frame to speedup recognition
                frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            self.face_data = self.get_recognized_face_data(frame, shrink_frames=shrink_frames)

            if self.count_fps:
                self.fps.update()

    def get_recognized_face_data(self, frame, shrink_frames=False):
        if self.database == {}:
            raise ValueError("Cannot recognize faces with empty database!")

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            return []
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_data = []
        for location, encoding in zip(face_locations, face_encodings):

            # get the distances from this encoding to those of all reference images
            distances = face_recognition.face_distance(self.known_face_encodings, encoding)

            # select the closest match (smallest distance) if it's below the threshold value
            if np.any(distances <= self.MAX_DISTANCE):
                best_match_idx = np.argmin(distances)
                identity = self.known_face_names[best_match_idx]
                logging.info("Face recognized: {}".format(identity))
            else:
                identity = None

            if shrink_frames:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                location = tuple(i*4 for i in location)

            face_data.append({
                'location': location,
                'location_format': 'top, right, bottom, left',
                'identity': identity
            })

        return face_data

    def get_fps(self):
        """
        Return elapsed fps of fps counter
        """
        if not self.count_fps:
            logging.error("No FPSCounter set")
            return None
        return self.fps.get_fps()

    def stop(self):
        self.stopped = True
        if self.count_fps:
            self.fps.stop()

    def __del__(self):
        self.stop()


############################## END CLASS ####################################


class ObjectRecognizer:

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    MODEL_PATH = './models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    LABELS_PATH = './src/object_detection/data/mscoco_label_map.pbtxt'

    NUM_CLASSES = 90

    def __init__(self, count_fps=False):
        self.graph = None  # tensorflow graph
        self.sess = None  # tensorflow session
        self.category_index = None
        self.stopped = False
        self.objects_data = []
        self.count_fps = count_fps
        if self.count_fps:
            self.fps = FPSCounter()

        self._load_label_categories()
        self._load_model()

    def _load_label_categories(self):
        # Loading label map
        label_map = label_map_util.load_labelmap(self.LABELS_PATH)
        categories = label_map_util.convert_label_map_to_categories(label_map=label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True
                                                                    )
        self.category_index = label_map_util.create_category_index(categories)

    def _load_model(self):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.MODEL_PATH, 'rb') as f:
                serialized_graph = f.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)
            self.graph = detection_graph

    def _detect_objects(self, image_np, shrink_frames=False):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        t_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        t_scores = self.graph.get_tensor_by_name('detection_scores:0')
        t_classes = self.graph.get_tensor_by_name('detection_classes:0')
        t_num_detections = self.graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [t_boxes, t_scores, t_classes, t_num_detections],
            feed_dict={image_tensor: image_np_expanded}
        )
        if shrink_frames:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            for i in range(num_detections):
                boxes[i] = tuple(i * 4 for i in boxes[i])

        return boxes, scores, classes, num_detections

    def read(self):
        return self.objects_data

    def start(self, frame_reader, shrink_frames=False):
        self.stopped = False
        threading.Thread(target=self._update, args=(frame_reader, shrink_frames)).start()

        if self.count_fps:
            self.fps.start()

    def _update(self, frame_reader, shrink_frames=False):
        while True:
            if self.stopped:
                logging.info("Object detection stopped. Quitting thread...")
                return

            ok, frame = frame_reader()
            if not ok:
                logging.info("ObjectsRecognizer could not read frame. Stopping...")
                return

            if shrink_frames:
                # shrink frame to speedup recognition
                frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            self.objects_data = (self._detect_objects(frame, shrink_frames=shrink_frames), self.category_index)

            if self.count_fps:
                self.fps.update()

    def get_fps(self):
        """
        Return elapsed fps of fps counter
        """
        if not self.count_fps:
            logging.error("No FPSCounter set")
            return None
        return self.fps.get_fps()

    def stop(self):
        self.stopped = True
        if self.count_fps:
            self.fps.stop()

    def __del__(self):
        self.stop()


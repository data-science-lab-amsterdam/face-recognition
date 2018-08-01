#########################################################
#
# Simple but faster version
#
#########################################################

import face_recognition
import cv2
import glob
import os
import numpy as np
import logging
from picamera.array import PiRGBArray
from picamera import PiCamera

IMAGES_PATH = './images'
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6


def setup_database():
    """
    Load anchor images and create a database of their face encodings
    """
    database = {}

    # load all the images of individuals to recognize into the database
    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
        # load image
        image = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0].split('-')[0]

        # detect faces and get the model encoding of the first face (there should be only 1 face in the anchor images)
        database[identity] = face_recognition.face_encodings(image)[0]

    return database


def run(database, resolution=(1280, 720), framerate=20):
    """
    Start the face recognition via the webcam
    """
    process_this_frame = True

    # Create arrays of known face encodings and their names
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())

    # initialize the camera and stream
    camera = PiCamera()
    camera.resolution = resolution
    camera.framerate = framerate
    raw_capture = PiRGBArray(camera, size=resolution)
    stream = camera.capture_continuous(raw_capture, format="bgr", use_video_port=True)

    for f in stream:
        try:
            # grab the frame from the stream and clear the stream in preparation for the next frame
            frame = f.array
            grabbed = True
            raw_capture.truncate(0)
        except Exception as e:
            logging.error("Something went wrong while reading from the camera")
            logging.error(e)
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # get the distances from this encoding to those of all reference images
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # select the closest match (smallest distance) if it's below the threshold value
                if np.any(distances <= MAX_DISTANCE):
                    best_match_idx = np.argmin(distances)
                    name = known_face_names[best_match_idx]
                else:
                    name = None

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    stream.close()
    raw_capture.close()
    camera.close()
    cv2.destroyAllWindows()


# run main script
database = setup_database()
run(database)

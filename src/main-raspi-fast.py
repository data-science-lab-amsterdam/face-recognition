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


def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings


def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}

    # load all the images of people to recognize into the database
    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
        # load image
        image_rgb = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0].split('-')[0]

        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        database[identity] = encodings[0]
        logging.info("Image of '{}' added to database".format(identity))

    return database


def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


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
                    logging.info("No match found. Smallest distance: {}".format(np.min(distances)))
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

            if name is None:
                name = 'Unknown'
                color = (0, 0, 255)  # red
            else:
                color = (0, 128, 0)  # dark green

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
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
logging.basicConfig(level=logging.INFO)
database = setup_database()
run(database)

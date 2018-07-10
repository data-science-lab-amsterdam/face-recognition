
# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import picamera
import numpy as np
import glob
import os

IMAGES_PATH = './images'

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
        # detect faces and get the model encoding of the first face
        database[identity] = face_recognition.face_encodings(image)[0]

    return database


def run(database):
    """
    Start the camera and run the detection
    """
    # Create arrays of known face encodings and their names
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())

    # Get a reference to the Raspberry Pi camera.
    # If this fails, make sure you have a camera connected to the RPi and that you
    # enabled your camera in raspi-config and rebooted first.
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    output = np.empty((240, 320, 3), dtype=np.uint8)

    while True:
        print("Capturing image")
        # Grab a single frame of video from the RPi camera as a numpy array
        camera.capture(output, format="rgb")

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(output)
        print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(output, face_locations)

        # Loop over each face found in the frame to see if it's someone we know.
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                name = "<Unknown Person>"

            print("I see someone named {}!".format(name))


# start program
database = setup_database()
run(database)

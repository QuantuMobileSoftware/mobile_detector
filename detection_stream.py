import argparse
from os import path
import time

import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

from object_detector import ObjectDetector


basepath = path.dirname(__file__)

if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser(prog='test_models.py')

    # add arguments
    parser.add_argument("--graph_path", "-gp", type=str, required=False,
                        default=path.join(basepath, "frozen_inference_graph.pb"),
                        help="path to image")

    # read arguments from the command line
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(k, v)

    # initialize detector
    print('model loading ...')
    predictor = ObjectDetector(args.graph_path)

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr",
                                           use_video_port=True):
        t1 = cv2.getTickCount()

        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        cv2.putText(image, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

        result = predictor.detect(np.expand_dims(image, axis=0))
        print(result)

        for object in result[0]:
            cv2.rectangle(image, object[0], object[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(object[3], object[2]),
                        (object[0][0], object[0][1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


        # show the frame
        cv2.imshow("Stream", image)
        key = cv2.waitKey(1) & 0xFF

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
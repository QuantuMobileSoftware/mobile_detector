import argparse
from os import path
import time
import logging
import sys
from enum import Enum

import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

from object_detector_detection_api import ObjectDetectorDetectionAPI
from yolo_darfklow import YOLODarkflowDetector


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger('detector')


class Models(Enum):
    ssd_light = 'ssd_light'
    tiny_yolo = 'tiny_yolo'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return Models[s]
        except KeyError:
            raise ValueError()

basepath = path.dirname(__file__)

if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser(prog='test_models.py')

    # add arguments
    parser.add_argument("--model_name", "-mn", type=str, required=True,
                        type=Models.from_string, choices=list(Models),
                        help="name of detection model: {}".format(list(Models)))
    parser.add_argument("--graph_path", "-gp", type=str, required=False,
                        default=path.join(basepath, "frozen_inference_graph.pb"),
                        help="path to ssdlight model frozen graph *.pb file")
    parser.add_argument("--cfg_path", "-cfg", type=str, required=False,
                        default=path.join(basepath, "tiny-yolo-voc.cfg"),
                        help="path to yolo *.cfg file")
    parser.add_argument("--weights_path", "-w", type=str, required=False,
                        default=path.join(basepath, "tiny-yolo-voc.weights"),
                        help="path to yolo weights *.weights file")

    # read arguments from the command line
    args = parser.parse_args()

    for k, v in vars(args).items():
        logger.info('Arguments. {}: {}'.format(k, v))

    # initialize detector
    logger.info('Model loading...')
    if args.model_name == Models.ssd_light:
        predictor = ObjectDetectorDetectionAPI(args.graph_path)
    elif args.model_name == Models.tiny_yolo:
        predictor = YOLODarkflowDetector(args.cfg_path, args.weights_path)

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

        logger.info("FPS: {0:.2f}".format(frame_rate_calc))
        cv2.putText(image, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

        result = predictor.detect(np.expand_dims(image, axis=0))

        for obj in result[0]:
            logger.info('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                        format(obj[0], obj[1], obj[3], obj[2]))

            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                        (obj[0][0], obj[0][1] - 5),
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
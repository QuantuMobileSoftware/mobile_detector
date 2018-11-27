import argparse
from os import path
import logging
import sys
import time

import numpy as np
import cv2

from utils.utils import load_image_into_numpy_array, Models
from object_detector_detection_api import ObjectDetectorDetectionAPI
from yolo_darfklow import YOLODarkflowDetector
from object_detector_detection_api_lite import ObjectDetectorLite


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger('detector')


basepath = path.dirname(__file__)


if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser(prog='test_models.py')

    # add arguments
    parser.add_argument("--image_path", "-ip", type=str, required=True,
                        help="path to image")
    parser.add_argument("--model_name", "-mn", type=Models.from_string,
                        required=True, choices=list(Models),
                        help="name of detection model: {}".format(
                        list(Models)))
    parser.add_argument("--cfg_path", "-cfg", type=str, required=False,
                        default=path.join(basepath, "tiny-yolo-voc.cfg"),
                        help="path to yolo *.cfg file")
    parser.add_argument("--graph_path", "-gp", type=str, required=False,
                        default=path.join(basepath, "frozen_inference_graph.pb"),
                        help="path to model frozen graph *.pb file")
    parser.add_argument("--result_path", "-rp", type=str, required=False,
                        default='result.jpg', help="path to result image")

    # read arguments from the command line
    args = parser.parse_args()

    for k, v in vars(args).items():
        logger.info('Arguments. {}: {}'.format(k, v))

    # initialize detector
    logger.info('Model loading...')
    if args.model_name == Models.ssd_lite:
        predictor = ObjectDetectorDetectionAPI(args.graph_path)
    # elif args.model_name == Models.tiny_yolo:
    #     predictor = YOLODarkflowDetector(args.cfg_path, args.weights_path)
    elif args.model_name == Models.tf_lite:
        predictor = ObjectDetectorLite()

    image = load_image_into_numpy_array(args.image_path)
    h, w, _ = image.shape

    start_time = time.time()
    result = predictor.detect(image)
    finish_time = time.time()
    logger.info("time spent: {:.4f}".format(finish_time - start_time))

    for obj in result:
        logger.info('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                    format(obj[0], obj[1], obj[3], obj[2]))

        cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
        cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                    (obj[0][0], obj[0][1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imwrite(args.result_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

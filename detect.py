import argparse
from os import path
import logging
import sys

import numpy as np
import cv2

from utils.utils import load_image_into_numpy_array
from object_detector_detection_api import ObjectDetector


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
    parser.add_argument("--graph_path", "-gp", type=str, required=False,
                        default=path.join(basepath, "frozen_inference_graph.pb"),
                        help="path to model frozen graph *.pb file")
    parser.add_argument("--result_path", "-rp", type=str, required=False,
                        default='result.jpg', help="path to result image")

    # read arguments from the command line
    args = parser.parse_args()

    for k, v in vars(args).items():
        logger.info('Arguments. {}: {}'.format(k, v))

    predictor = ObjectDetector(args.graph_path)

    image = load_image_into_numpy_array(args.image_path)
    h, w, _ = image.shape

    result = predictor.detect(image)

    for obj in result[0]:
        logger.info('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                    format(obj[0], obj[1], obj[3], obj[2]))

        cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
        cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                    (obj[0][0], obj[0][1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imwrite(args.result_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    predictor.close()

import argparse
from os import path
import time
import logging
import sys
import cv2

from object_detector_detection_api_lite import ObjectDetectorLite
from object_detector_detection_api import ObjectDetectorDetectionAPI


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger('detector')


basepath = path.dirname(__file__)


def open_cam_onboard(width, height):
    # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
    gst_str = ('nvcamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int)2592, height=(int)1458, '
               'format=(string)I420, framerate=(fraction)30/1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser(prog='test_models.py')

    # add arguments
    parser.add_argument("--graph_path", "-gp", type=str, required=False,
                        default=path.join(basepath,
                                          "frozen_inference_graph.pb"),
                        help="path to ssdlight model frozen graph *.pb file")

    # read arguments from the command line
    args = parser.parse_args()

    # initialize detector
    logger.info('Model loading...')
    predictor = ObjectDetectorDetectionAPI(args.graph_path)

    cap = open_cam_onboard(640, 480)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    # allow the camera to warmup
    time.sleep(0.1)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while (cap.isOpened()):
        t1 = cv2.getTickCount()

        ret, frame = cap.read()

        logger.info("FPS: {0:.2f}".format(frame_rate_calc))
        cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

        result = predictor.detect(frame)

        for obj in result:
            logger.info('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                        format(obj[0], obj[1], obj[3], obj[2]))

            cv2.rectangle(frame, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(frame, '{}: {:.2f}'.format(obj[3], obj[2]),
                        (obj[0][0], obj[0][1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # show the frame
        cv2.imshow("Stream", frame)
        key = cv2.waitKey(1) & 0xFF

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    predictor.close()

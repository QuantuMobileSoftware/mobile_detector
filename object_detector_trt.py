import numpy as np
import tensorflow as tf
import cv2
import tensorflow.contrib.tensorrt as trt
from tf_trt_models.detection import download_detection_model, \
    build_detection_graph

from object_detector_detection_api import ObjectDetectorDetectionAPI, \
    PATH_TO_LABELS, NUM_CLASSES


CONFIG_PATH = 'data/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config'
CHECKPOINT_PATH = 'data/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt'

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class ObjectDetectorTRT(ObjectDetectorDetectionAPI):
    def __init__(self, config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH):
        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        frozen_graph, input_names, output_names = build_detection_graph(
            config=config_path,
            checkpoint=checkpoint_path,
            score_threshold=0.3,
            batch_size=1
        )

        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=50
        )

        self.tf_sess = tf.Session(config=tf_config)

        tf.import_graph_def(trt_graph, name='')

        self.tf_input = \
            self.tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
        self.tf_scores =\
            self.tf_sess.graph.get_tensor_by_name('detection_scores:0')
        self.tf_boxes = \
            self.tf_sess.graph.get_tensor_by_name('detection_boxes:0')
        self.tf_classes = \
            self.tf_sess.graph.get_tensor_by_name('detection_classes:0')
        self.tf_num_detections = \
            self.tf_sess.graph.get_tensor_by_name('num_detections:0')

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """
        image_resized = cv2.resize(image, (300, 300))
        image_resized = np.expand_dims(image_resized, axis=0)

        scores, boxes, classes, num_detections = self.tf_sess.run(
            [
                self.tf_scores,
                self.tf_boxes,
                self.tf_classes,
                self.tf_num_detections
            ],
            feed_dict={self.tf_input: image_resized}
        )

        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(boxes),
                            np.squeeze(classes+1).astype(np.int32),
                            np.squeeze(scores),
                            min_score_thresh=threshold)

    def close(self):
        self.tf_sess.close()

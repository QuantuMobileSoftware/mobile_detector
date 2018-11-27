import numpy as np
import tensorflow as tf
import cv2
import time

from object_detector import ObjectDetector


INPUT_MEAN = 127.5
INPUT_STD = 127.5
ALT_OUTPUT_ORDER = False
MAX_BOXES = 10

NUM_RESULTS = 1917
NUM_CLASSES = 91

X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0


class ObjectDetectorLite(ObjectDetector):
    def __init__(self, model_file='mobilenet_ssd.tflite',
                 box_prior_file='data/box_priors.txt',
                 label_file='data/coco_labels.txt'):
        self.interpreter = tf.contrib.lite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        if self.input_details[0]['dtype'] == type(np.float32(1.0)):
            self.floating_model = True
        else:
            self.floating_model = False

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.box_priors = []
        self.load_box_priors(box_prior_file)
        self.labels = self.load_labels(label_file)

    def detect(self, frame, threshold=0.1):
        h, w, _ = frame.shape
        w_ratio = w / self.width
        h_ratio = h / self.height

        frame = cv2.resize(frame, (self.width, self.height), cv2.INTER_LINEAR)
        # add N dim
        input_data = np.expand_dims(frame, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        start_time = time.time()
        self.interpreter.invoke()
        finish_time = time.time()
        print("time spent: {:.4f}".format(finish_time - start_time))

        p_index = 0
        o_index = 1
        if ALT_OUTPUT_ORDER:
            p_index = 1
            o_index = 0

        predictions = np.squeeze( \
            self.interpreter.get_tensor(self.output_details[p_index]['index']))
        output_classes = np.squeeze( \
            self.interpreter.get_tensor(self.output_details[o_index]['index']))
        if not self.floating_model:
            p_scale, p_mean = self.output_details[p_index]['quantization']
            o_scale, o_mean = self.output_details[o_index]['quantization']

            predictions = (predictions - p_mean * 1.0) * p_scale
            output_classes = (output_classes - o_mean * 1.0) * o_scale

        self.decode_center_size_boxes(predictions)

        pruned_predictions = [[], ]
        for c in range(1, NUM_CLASSES):
            pruned_predictions.append([])
            for r in range(0, NUM_RESULTS):
                score = 1. / (1. + np.exp(-output_classes[r][c]))
                output_classes[r][c] = score
                if score > 0.2:
                    rect = (
                    predictions[r][1] * self.width, predictions[r][0] * self.width, \
                    predictions[r][3] * self.width, predictions[r][2] * self.width)

                    pruned_predictions[c].append(
                        (output_classes[r][c], r, self.labels[c], rect))

        final_predictions = []
        for c in range(1, NUM_CLASSES):
            predictions_for_class = pruned_predictions[c]
            suppressed_predictions = self.nms(predictions_for_class, 0.3, MAX_BOXES)
            final_predictions = final_predictions + suppressed_predictions

        final_predictions = sorted(final_predictions, reverse=True)[:MAX_BOXES]

        final_predictions = [[(int(i[3][0] * w_ratio), int(i[3][1] * h_ratio)),
                              (int(i[3][2] * w_ratio), int(i[3][3] * h_ratio)),
                              i[0], i[2]]
                             for i in final_predictions]
        return final_predictions

    def load_box_priors(self, filename):
        with open(filename) as f:
            count = 0
            for line in f:
                row = line.strip().split(' ')
                self.box_priors.append(row)
                # print(box_priors[count][0])
                count = count + 1
                if count == 4:
                    return

    def load_labels(self, filename):
        my_labels = []
        input_file = open(filename, 'r')
        for l in input_file:
            my_labels.append(l.strip())
        return my_labels

    def decode_center_size_boxes(self, locations):
        """calculate real sizes of boxes"""
        for i in range(0, NUM_RESULTS):
            ycenter = locations[i][0] / Y_SCALE * np.float(self.box_priors[2][i]) \
                      + np.float(self.box_priors[0][i])
            xcenter = locations[i][1] / X_SCALE * np.float(self.box_priors[3][i]) \
                      + np.float(self.box_priors[1][i])
            h = np.exp(locations[i][2] / H_SCALE) * np.float(
                self.box_priors[2][i])
            w = np.exp(locations[i][3] / W_SCALE) * np.float(
                self.box_priors[3][i])

            ymin = ycenter - h / 2.0
            xmin = xcenter - w / 2.0
            ymax = ycenter + h / 2.0
            xmax = xcenter + w / 2.0

            locations[i][0] = ymin
            locations[i][1] = xmin
            locations[i][2] = ymax
            locations[i][3] = xmax
        return locations

    @staticmethod
    def iou(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        intersection_area = (x_b - x_a + 1) * (y_b - y_a + 1)

        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        iou = intersection_area / float(
            box_a_area + box_b_area - intersection_area)
        return iou

    def nms(self, p, iou_threshold, max_boxes):
        sorted_p = sorted(p, reverse=True)
        selected_predictions = []
        for a in sorted_p:
            if len(selected_predictions) > max_boxes:
                break
            should_select = True
            for b in selected_predictions:
                if self.iou(a[3], b[3]) > iou_threshold:
                    should_select = False
                    break
            if should_select:
                selected_predictions.append(a)

        return selected_predictions


if __name__ == '__main__':
    detector = ObjectDetectorLite()

    image = cv2.cvtColor(cv2.imread('dog.jpg'), cv2.COLOR_BGR2RGB)

    result = detector.detect(image)
    print(result)

    for obj in result:

        cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
        cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                    (obj[0][0], obj[0][1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imwrite('resres.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

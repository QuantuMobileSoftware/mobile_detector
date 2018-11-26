# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""objection_detection for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import time
from heapq import heappush, nlargest

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# from lite import interpreter as interpreter_wrapper

import tensorflow as tf

NUM_RESULTS = 1917
NUM_CLASSES = 91

X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

def load_box_priors(filename):
  with open(filename) as f:
    count = 0
    for line in f:
      row = line.strip().split(' ')
      box_priors.append(row)
      #print(box_priors[count][0])
      count = count + 1
      if count == 4:
        return

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

def decode_center_size_boxes(locations):
  """calculate real sizes of boxes"""
  for i in range(0, NUM_RESULTS):
    ycenter = locations[i][0] / Y_SCALE * np.float(box_priors[2][i]) \
            + np.float(box_priors[0][i])
    xcenter = locations[i][1] / X_SCALE * np.float(box_priors[3][i]) \
            + np.float(box_priors[1][i])
    h = math.exp(locations[i][2] / H_SCALE) * np.float(box_priors[2][i])
    w = math.exp(locations[i][3] / W_SCALE) * np.float(box_priors[3][i])

    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0

    locations[i][0] = ymin
    locations[i][1] = xmin
    locations[i][2] = ymax
    locations[i][3] = xmax
  return locations

def iou(box_a, box_b):
  x_a = max(box_a[0], box_b[0])
  y_a = max(box_a[1], box_b[1])
  x_b = min(box_a[2], box_b[2])
  y_b = min(box_a[3], box_b[3])

  intersection_area = (x_b - x_a + 1) * (y_b - y_a + 1)

  box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
  box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

  iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
  return iou

def nms(p, iou_threshold, max_boxes):
  sorted_p = sorted(p, reverse=True)
  selected_predictions = []
  for a in sorted_p:
    if len(selected_predictions) > max_boxes:
      break
    should_select = True
    for b in selected_predictions:
      if iou(a[3], b[3]) > iou_threshold:
        should_select = False
        break
    if should_select:
      selected_predictions.append(a)

  return selected_predictions

if __name__ == "__main__":
  file_name = "dog.jpg"
  model_file = "mobilenet_ssd.tflite"
  label_file = "data/coco_labels.txt"
  box_prior_file = "data/box_priors.txt"
  input_mean = 127.5
  input_std = 127.5
  min_score = 20.0
  max_boxes = 10
  floating_model = False
  show_image = False
  alt_output_order = False

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be classified")
  parser.add_argument("--graph", help=".tflite model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_mean", help="input_mean")
  parser.add_argument("--input_std", help="input standard deviation")
  parser.add_argument("--min_score", help="show only > min_score")
  parser.add_argument("--max_boxes", help="max boxes to show")
  parser.add_argument("--show_image", help="show image", default=True)
  parser.add_argument("--alt_output_order", help="alternative output index")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_mean:
    input_mean = float(args.input_mean)
  if args.input_std:
    input_std = float(args.input_std)
  if args.min_score:
    min_score = float(args.min_score)
  if args.max_boxes:
    max_boxes = int(args.max_boxes)
  if args.show_image:
    show_image = args.show_image
  if args.alt_output_order:
    alt_output_order = args.alt_output_order

  interpreter = tf.contrib.lite.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  #print(input_details)
  #print(output_details)

  # check the type of the input tensor
  if input_details[0]['dtype'] == type(np.float32(1.0)):
    floating_model = True

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(file_name)
  img = img.resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  finish_time = time.time()
  print("time spent:", ((finish_time - start_time) * 1000))

  box_priors = []
  load_box_priors(box_prior_file)
  labels = load_labels(label_file)

  p_index = 0
  o_index = 1
  if alt_output_order:
    p_index = 1
    o_index = 0

  predictions = np.squeeze( \
                  interpreter.get_tensor(output_details[p_index]['index']))
  output_classes = np.squeeze( \
                     interpreter.get_tensor(output_details[o_index]['index']))
  if not floating_model:
    p_scale, p_mean = output_details[p_index]['quantization']
    o_scale, o_mean = output_details[o_index]['quantization']

    predictions = (predictions - p_mean * 1.0) * p_scale
    output_classes = (output_classes - o_mean * 1.0) * o_scale

  decode_center_size_boxes(predictions)

  pruned_predictions = [[],]
  for c in range(1, NUM_CLASSES):
    pruned_predictions.append([])
    for r in range(0, NUM_RESULTS):
      score = 1. / (1. + math.exp(-output_classes[r][c]))
      if score > 0.01:
        rect = (predictions[r][1] * width, predictions[r][0] * width, \
                predictions[r][3] * width, predictions[r][2] * width)

        pruned_predictions[c].append((output_classes[r][c], r, labels[c], rect))

  final_predictions = []
  for c in range(1, NUM_CLASSES):
    predictions_for_class = pruned_predictions[c]
    suppressed_predictions = nms(predictions_for_class, 0.5, max_boxes)
    final_predictions = final_predictions +  suppressed_predictions

  if show_image:
    fig, ax = plt.subplots(1)

  final_predictions = sorted(final_predictions, reverse=True)[:max_boxes]
  for e in final_predictions:
    score = 100. / (1. + math.exp(-e[0]))
    score_string = '{0:2.0f}%'.format(score)
    print(score_string, e[2], e[3])
    if score < min_score:
      break
    left, top, right, bottom = e[3]
    rect = patches.Rectangle((left, top), (right - left), (bottom - top), \
             linewidth=1, edgecolor='r', facecolor='none')

    if show_image:
      # Add the patch to the Axes
      ax.add_patch(rect)
      ax.text(left, top, e[2]+': '+score_string, fontsize=6,
              bbox=dict(facecolor='y', edgecolor='y', alpha=0.5))

  if show_image:
    ax.imshow(img)
    plt.title(model_file)
    plt.show()
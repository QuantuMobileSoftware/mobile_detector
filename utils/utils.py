import numpy as np
import cv2
from enum import Enum


class Models(Enum):
    ssd_lite = 'ssd_lite'
    tiny_yolo = 'tiny_yolo'
    tf_lite = 'tf_lite'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return Models[s]
        except KeyError:
            raise ValueError()


MAX_AREA = 0.019  # max area from train set
RATIO_MEAN = 4.17
RATIO_STD = 1.06


def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def affine_tile_corners(x0, y0, theta, wp, hp):
    """
    Find corners of tile defined by affine transformation.

    Find corners in original image for tile defined by affine transformation,
    i.e. a rotation and translation, given (x0, y0) the upper left corner of
    the tile, theta, the rotation angle of the tile in degrees, and the tile
    width wp, and height hp.

    Args:
        x0          Horizontal coordinate of tile upper left corner (pixels)
        y0          Vertical coordinate of tile upper left corner (pixels)
        theta       Rotation angle (degrees clockwise from vertical)
        wp          Tile width (pixels)
        hp          Tile height (pixels)
    Returns:
        corners     Corner points, in clockwise order starting from upper left
                    corner, ndarray size (4, 2)
    """
    rot_angle = np.radians(theta)
    corners = np.array(
        [[x0, y0],
         [x0 + wp * np.cos(rot_angle), y0 + wp * np.sin(rot_angle)],
         [x0 + wp * np.cos(rot_angle) - hp * np.sin(rot_angle),
          y0 + wp * np.sin(rot_angle) + hp * np.cos(rot_angle)],
         [x0 - hp * np.sin(rot_angle), y0 + hp * np.cos(rot_angle)]])
    return corners


def tile_images(tiling_params, img):
    res = []
    original_sizes = []
    offset = []

    for cur_pt, cur_theta, cur_multiplier in zip(
            tiling_params["upper_left_pts"],
            tiling_params["thetas"],
            tiling_params["multipliers"]):
        cur_x0, cur_y0 = cur_pt
        corners = affine_tile_corners(
            cur_x0, cur_y0, cur_theta,
            int(cur_multiplier * tiling_params["wp"]),
            int(cur_multiplier * tiling_params["hp"])).astype(int)

        top = min(corners[:, 1])
        left = min(corners[:, 0])
        bottom = max(corners[:, 1])
        right = max(corners[:, 0])
        h = bottom - top
        w = right - left
        tile = np.zeros((h, w, 3)).astype(np.uint8)

        # crop tile from image
        tmp = img[top: bottom, left: right]
        tile[:tmp.shape[0], :tmp.shape[1], :3] = tmp

        # resize the tile
        tile = cv2.resize(tile, (tiling_params["wp"], tiling_params["hp"]),
                          interpolation=cv2.INTER_NEAREST)

        # rotate the tile
        image_center = tuple(np.array(tile.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, cur_theta, 1.0)
        tmp = cv2.warpAffine(tile, rot_mat, (tile.shape[1::-1]),
                             flags=cv2.INTER_LINEAR)

        original_sizes.append((bottom - top, right - left))
        offset.append((top, left))
        res.append(tmp)

    return res, original_sizes, offset


def rotate_points(points, rotation_matrix):
    # add ones
    points_ones = np.append(points, 1)

    # transform points
    transformed_points = rotation_matrix.dot(points_ones)
    return transformed_points# [:,::-1]


def split_img(img, m, n):
    h, w, _ = img.shape
    tile_h = h // m
    tile_w = w // n
    padding_h = tile_h // 10
    padding_w = int(tile_w * 0.15)

    res = []
    original_sizes = []
    offset = []
    for i in range(0, m):
        top = i * tile_h
        bottom = min(h, (i + 1) * tile_h + padding_h)
        for j in range(0, n):
            left = j * tile_w
            right = min(w, (j + 1) * tile_w + padding_w)
            original_sizes.append((bottom - top, right - left))
            offset.append((top, left))
            res.append(cv2.resize(img[top: bottom, left: right, :],
                                  (tile_w, tile_h),
                                  interpolation=cv2.INTER_NEAREST))

    return res, original_sizes, offset


def get_global_coord(point, img_size, original_size, offset):
    return [int(point[0] / img_size[1] * original_size[1] + offset[1]), \
           int(point[1] / img_size[0] * original_size[0] + offset[0])]


def non_max_suppression_fast(boxes, labels, overlap_thresh=0.5):
    # if there are no boxes, return an empty list
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return [], []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = 1. * (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], [labels[i] for i in pick]


def filter_bb_by_size(bbs, labels, img_area):
    res_bbs = []
    res_labels = []
    for bb, l in zip(bbs, labels):
        s = (bb[2] - bb[0]) * (bb[3] - bb[1]) / img_area
        r = (bb[3] - bb[1]) / (bb[2] - bb[0])
        if s < MAX_AREA * 1.1 and RATIO_MEAN - 3 * RATIO_MEAN < r < RATIO_MEAN + 3 * RATIO_MEAN:
            res_bbs.append(bb)
            res_labels.append(l)

    return res_bbs, res_labels

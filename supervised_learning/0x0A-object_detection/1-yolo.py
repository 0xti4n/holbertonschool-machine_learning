#!/usr/bin/env python3
"""yolo V3"""
import tensorflow as tf
import numpy as np


class Yolo():
    """class Yolo that uses the Yolo v3
    algorithm to perform object detection

    -> model_path is the path to where a Darknet Keras model is stored

    -> classes_path is the path to where the list of class names used
    for the Darknet model, listed in order of index, can be found

    -> class_t is a float representing the box score threshold
    for the initial filtering step

    -> nms_t is a float representing the IOU threshold
    for non-max suppression

    -> anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
    containing all of the anchor boxes:
            outputs is the number of outputs (predictions)
            made by the Darknet model
            anchor_boxes is the number of anchor boxes
            used for each prediction
            2 => [anchor_box_width, anchor_box_height]
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            lines = f.readlines()
        self.class_names = [line.split('\n')[0] for line in lines]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ Process Outputs

        -> outputs is a list of numpy.ndarrays containing
        the predictions from the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                * grid_height & grid_width => the height and width of
                the grid used for the output
                * anchor_boxes => the number of anchor boxes used
                * 4 => (t_x, t_y, t_w, t_h)
                * 1 => box_confidence
                * classes => class probabilities for all classes

        -> image_size is a numpy.ndarray containing the image’s original
        size [image_height, image_width]

        -> Returns a tuple of (boxes, box_confidences, box_class_probs):
            * boxes: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 4) containing the processed
            boundary boxes for each output, respectively:
                - 4 => (x1, y1, x2, y2)
                -(x1, y1, x2, y2) should represent the boundary
                box relative to original image

            * box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing
            the box confidences for each output, respectively

            * box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing
            the box’s class probabilities for each output, respectively"""

        boxes = []
        box_confidences = []
        box_class_probs = []
        H = image_size[0]
        W = image_size[1]

        def sigmoid(X):
            """sigmoid function
            """
            return 1.0 / (1.0 + np.exp(-X))

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        for i, box in enumerate(boxes):
            g_H = box.shape[0]
            g_W = box.shape[1]
            anchor_boxes = box.shape[2]
            grid = np.zeros((g_H, g_W, anchor_boxes))
            I_y = np.arange(g_H).reshape(g_H, 1, 1)
            I_x = np.arange(g_W).reshape(1, g_W, 1)
            c_y = grid + I_y
            c_x = grid + I_x
            t_x = sigmoid(box[..., 0])
            t_y = sigmoid(box[..., 1])
            t_w = box[..., 2]
            t_h = box[..., 3]
            bx = t_x + c_x
            by = t_y + c_y
            bx = bx / g_W
            by = by / g_H
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)
            modelInput_w = self.model.input.shape[1].value
            modelInput_h = self.model.input.shape[2].value
            bw = bw / modelInput_w
            bh = bh / modelInput_h
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            box[..., 0] = x1 * W
            box[..., 1] = y1 * H
            box[..., 2] = x2 * W
            box[..., 3] = y2 * H
        items = (boxes, box_confidences, box_class_probs)
        return items

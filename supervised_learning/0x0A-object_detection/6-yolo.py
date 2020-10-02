#!/usr/bin/env python3
"""yolo V3"""
import tensorflow as tf
import numpy as np
import glob
import cv2
import os


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter Boxes

        -> boxes: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4) containing the
        processed boundary boxes for each output, respectively

        -> box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1) containing the
        processed box confidences for each output, respectively

        -> box_class_probs: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, classes) containing
        the processed box class probabilities for each output, respectively

        ->Returns a tuple of (filtered_boxes, box_classes, box_scores):
            * filtered_boxes: a numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes:
            * box_classes: a numpy.ndarray of shape (?,)
            containing the class number that each box in
            filtered_boxes predicts, respectively
            * box_scores: a numpy.ndarray of shape (?)
            containing the box scores for each box in
            filtered_boxes, respectively
        """
        f_box = []
        b_class = []
        b_clas_score = []

        for i in range(len(boxes)):
            box_scores = box_confidences[i] * box_class_probs[i]

            box_class_scores = np.max(box_scores, axis=-1).reshape(-1)
            del_idx = np.where(box_class_scores < self.class_t)
            box_class_scores = np.delete(box_class_scores, del_idx)
            b_clas_score.append(box_class_scores)
            box_classes = np.argmax(box_scores, axis=-1).reshape(-1)
            box_classes = np.delete(box_classes, del_idx)
            b_class.append(box_classes)
            box = boxes[i].reshape(-1, 4)
            filter_box = np.delete(box, del_idx, axis=0)
            f_box.append(filter_box)

        f = np.concatenate(f_box)
        clas = np.concatenate(b_class)
        b_c_s = np.concatenate(b_clas_score)

        return f, clas, b_c_s

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """non_max suppression"""
        index = np.lexsort((-box_scores, box_classes))
        box_predictions = filtered_boxes[index]
        predict_box_classes = box_classes[index]
        predict_box_scores = box_scores[index]
        _, n_classes = np.unique(predict_box_classes,
                                 return_counts=True)

        i = 0
        con = 0
        for n_class in n_classes:
            while i < con + n_class:
                j = i + 1
                while j < con + n_class:
                    b_p1 = box_predictions[i]
                    b_p2 = box_predictions[j]
                    x_x1 = max(b_p1[0], b_p2[0])
                    y_y1 = max(b_p1[1], b_p2[1])
                    x_x2 = min(b_p1[2], b_p2[2])
                    y_y2 = min(b_p1[3], b_p2[3])

                    inter_area = max(y_y2 - y_y1, 0) * max(x_x2 - x_x1, 0)

                    box1_area = (b_p1[3] - b_p1[1]) * (b_p1[2] - b_p1[0])
                    box2_area = (b_p2[3] - b_p2[1]) * (b_p2[2] - b_p2[0])
                    union_area = box1_area + box2_area - inter_area

                    iou = inter_area / union_area

                    if iou > self.nms_t:
                        box_predictions = np.delete(box_predictions,
                                                    j, axis=0)
                        predict_box_scores = np.delete(predict_box_scores,
                                                       j, axis=0)
                        predict_box_classes = np.delete(predict_box_classes,
                                                        j, axis=0)
                        n_class -= 1
                    else:
                        j += 1
                i += 1
            con += n_class
        return box_predictions, predict_box_classes, predict_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images
        """
        img = glob.glob(folder_path + "/*")
        data_p = [cv2.imread(i) for i in img]

        return data_p, img

    def preprocess_images(self, images):
        """ Preprocess images """
        images_shape = []
        list_resize = []
        modelInput_w = self.model.input.shape[1].value
        modelInput_h = self.model.input.shape[2].value

        for img in images:
            shape = img.shape[:2]
            images_shape.append(shape)
            img_resize = cv2.resize(img, (modelInput_w, modelInput_h),
                                    interpolation=cv2.INTER_CUBIC)

            scale = img_resize / 255
            list_resize.append(scale)

        list_resize = np.array(list_resize)
        images_shape = np.array(images_shape)

        return list_resize, images_shape

    def show_boxes(self, image,
                   boxes, box_classes,
                   box_scores, file_name):
        """Show boxes"""
        for i in range(len(boxes)):
            x = int(boxes[i][0])
            y = int(boxes[i][1])
            w = int(boxes[i][2])
            h = int(boxes[i][3])
            score = str(round(box_scores[i]))
            class_name = str(self.class_names[box_classes[i]])
            label = class_name + " " + score
            color = (255, 0, 0)
            color1 = (0, 0, 255)
            cv2.rectangle(image, (x, y), (w, h), color, 2)
            cv2.putText(image, label, (x - 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color1, 1, cv2.LINE_AA)
        cv2.imshow(file_name, image)
        k = cv2.waitKey(0)

        if k == ord('s'):
            dir_name = 'detections'
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            cv2.imwrite(os.path.join(dir_name, file_name), image)
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()

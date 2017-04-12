import init_path
import caffe
import cv2
import os
import numpy as np
from utils import *
import time


class FaceNet(object):

    def __init__(self, caffe_model_path, device_id):
        caffe.set_mode_cpu()
        caffe.set_device(device_id)
        self._PNet = caffe.Net(
            os.path.join(caffe_model_path, 'det1.prototxt'),
            os.path.join(caffe_model_path, 'det1.caffemodel'),
            caffe.TEST)
        self._RNet = caffe.Net(
            os.path.join(caffe_model_path, 'det2.prototxt'),
            os.path.join(caffe_model_path, 'det2.caffemodel'),
            caffe.TEST)
        self._ONet = caffe.Net(
            os.path.join(caffe_model_path, 'det3.prototxt'),
            os.path.join(caffe_model_path, 'det3.caffemodel'),
            caffe.TEST)

        self._VerifyNet = caffe.Net(
            os.path.join(caffe_model_path, 'verify.prototxt'),
            os.path.join(caffe_model_path, 'verify.caffemodel'),
            caffe.TEST)
        input_shape = self._VerifyNet.blobs['data'].data.shape
        transformer = caffe.io.Transformer({'data': input_shape})
        if self._VerifyNet.blobs['data'].data.shape[1] == 3:
            transformer.set_transpose('data', (2, 0, 1))
        self._transformer = transformer
        self._sample_shape = self._VerifyNet.blobs['data'].data.shape

        # left eye, right eye, mouse center((left lip + right lip)/2)
        self._mean_pose = np.array(
            [(70.7450, 112.0000),
                (108.2370, 112.0000),
                (89.4324, 153.5140)],
            np.float32)

    def detect_face(
            self, img_in, minsize=20, threshold=[0.6, 0.7, 0.7],
            fastresize=False, factor=0.709):
        """
        face detection and alignment
        param:
            img_in: input image
            minsize: minsize of the bbox
            threshold: a list of three thresholds for the three stages
            fastresize: bool
            factor: pyramid factor
        return:
            bboxes: (face_num, 5);
                    x_lt, y_lt, x_rb, y_rb, prob
            landmarks: (face_num, 5, 2);
                    left_eye, right_eye, nose, left_lip, right_lip
        """
        img = img_in.copy()
        img = img[:, :, [2, 1, 0]]  # change BGR to RGB
        factor_count = 0
        total_boxes = np.zeros((0, 9), np.float)
        points = []
        h = img.shape[0]
        w = img.shape[1]
        minl = min(h, w)
        img = img.astype(float)
        m = 12.0/minsize
        minl = minl*m

        # create scale pyramid
        scales = []
        while minl >= 12:
            scales.append(m * pow(factor, factor_count))
            minl *= factor
            factor_count += 1

        # first stage
        for scale in scales:
            hs = int(np.ceil(h*scale))
            ws = int(np.ceil(w*scale))
            if fastresize:
                im_data = (img-127.5)*0.0078125  # [0,255] -> [-1,1]
                im_data = cv2.resize(im_data, (ws, hs))  # default is bilinear
            else:
                im_data = cv2.resize(img, (ws, hs))  # default is bilinear
                im_data = (im_data-127.5)*0.0078125  # [0,255] -> [-1,1]
            im_data = np.swapaxes(im_data, 0, 2)
            im_data = np.array([im_data], dtype=np.float)
            self._PNet.blobs['data'].reshape(1, 3, ws, hs)
            self._PNet.blobs['data'].data[...] = im_data
            out = self._PNet.forward()
            boxes = generateBoundingBox(
                out['prob1'][0, 1, :, :],
                out['conv4-2'][0],
                scale,
                threshold[0])
            if boxes.shape[0] != 0:
                pick = nms(boxes, 0.5, 'Union')

                if len(pick) > 0:
                    boxes = boxes[pick, :]
            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # nms
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            # revise and convert to square
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            total_boxes[:, 0] += total_boxes[:, 5]*regw
            total_boxes[:, 1] += total_boxes[:, 6]*regh
            total_boxes[:, 2] += total_boxes[:, 7]*regw
            total_boxes[:, 3] += total_boxes[:, 8]*regh
            total_boxes = rerec(total_boxes)
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes, w, h)

        #####
        # 1 #
        #####
        # print "[1]:", total_boxes.shape

        # second stage
        numbox = total_boxes.shape[0]
        if numbox > 0:
            # construct input for RNetConvolution
            tempimg = np.zeros((numbox, 24, 24, 3))  # (24, 24, 3, numbox)
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
                tempimg[k, :, :, :] = cv2.resize(tmp, (24, 24))
            tempimg = (tempimg-127.5)*0.0078125
            # RNet
            tempimg = np.swapaxes(tempimg, 1, 3)
            self._RNet.blobs['data'].reshape(numbox, 3, 24, 24)
            self._RNet.blobs['data'].data[...] = tempimg
            out = self._RNet.forward()
            score = out['prob1'][:, 1]
            pass_t = np.where(score > threshold[1])[0]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
            mv = out['conv5-2'][pass_t, :].T
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes, 0.7, 'Union')
                if len(pick) > 0:
                    total_boxes = total_boxes[pick, :]
                    total_boxes = bbreg(total_boxes, mv[:, pick])
                    total_boxes = rerec(total_boxes)
            #####
            # 2 #
            #####
            # print "[2]:", total_boxes.shape

            # third stage
            numbox = total_boxes.shape[0]
            if numbox > 0:
                total_boxes = np.fix(total_boxes)
                dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes, w, h)
                tempimg = np.zeros((numbox, 48, 48, 3))
                for k in range(numbox):
                    tmp = np.zeros((tmph[k], tmpw[k], 3))
                    tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
                    tempimg[k, :, :, :] = cv2.resize(tmp, (48, 48))
                tempimg = (tempimg-127.5)*0.0078125  # [0,255] -> [-1,1]
                # ONet
                tempimg = np.swapaxes(tempimg, 1, 3)
                self._ONet.blobs['data'].reshape(numbox, 3, 48, 48)
                self._ONet.blobs['data'].data[...] = tempimg
                out = self._ONet.forward()
                score = out['prob1'][:, 1]
                points = out['conv6-3']
                pass_t = np.where(score > threshold[2])[0]
                points = points[pass_t, :]
                score = np.array([score[pass_t]]).T
                total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
                mv = out['conv6-2'][pass_t, :].T
                w = total_boxes[:, 3] - total_boxes[:, 1] + 1
                h = total_boxes[:, 2] - total_boxes[:, 0] + 1
                points[:, 0:5] = np.tile(w, (5, 1)).T * points[:, 0:5] + \
                    np.tile(total_boxes[:, 0], (5, 1)).T - 1
                points[:, 5:10] = np.tile(h, (5, 1)).T * points[:, 5:10] + \
                    np.tile(total_boxes[:, 1], (5, 1)).T - 1
                if total_boxes.shape[0] > 0:
                    total_boxes = bbreg(total_boxes, mv)
                    pick = nms(total_boxes, 0.7, 'Min')
                    if len(pick) > 0:
                        total_boxes = total_boxes[pick, :]
                        points = points[pick, :]

        #####
        # 3 #
        #####
        # print "[3]:", total_boxes.shape

        face_num = total_boxes.shape[0]
        bboxes = total_boxes.copy()
        landmarks = np.zeros([face_num, 5, 2])
        for i in range(face_num):
            for j in range(5):
                landmarks[i, j, 0] = points[i, j]
                landmarks[i, j, 1] = points[i, j+5]

        return bboxes, landmarks

    def extract_feature(
            self, image, score_name,
            crop_size=(110, 110),
            image_size=(256, 256)):
        """
        extract verification feature of one face
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if crop_size is not None:
            center_x = image.shape[0]/2 + 15
            center_y = image.shape[1]/2
            w = crop_size[0]
            h = crop_size[1]
            image = image[center_x-w/2:center_x+w/2, center_y-h/2:center_y+h/2, :]
        if image_size is not None:
            image = cv2.resize(image, image_size)
        data = np.array([self._transformer.preprocess('data', image)])
        data = data*3.2/255-1.6  # scale data
        self._VerifyNet.blobs['data'].reshape(*data.shape)
        self._VerifyNet.reshape()
        out = self._VerifyNet.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()

    def extract_feature_batch(
        self, image_list, score_name,
        crop_size=(110, 110),
            image_size=(256, 256)):
        """
        extract verification feature of faces
        """
        new_image_list = []
        for image in image_list:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if crop_size is not None:
                center_x = image.shape[0]/2 + 15
                center_y = image.shape[1]/2
                w = crop_size[0]
                h = crop_size[1]
                image = image[center_x-w/2:center_x+w/2, center_y-h/2:center_y+h/2, :]
            if image_size is not None:
                image = cv2.resize(image, image_size)
            new_image_list.append(image.copy())
        data = np.array([self._transformer.preprocess('data', image)
                        for image in new_image_list])
        data = data*3.2/255-1.6  # scale data
        self._VerifyNet.blobs['data'].reshape(*data.shape)
        self._VerifyNet.reshape()
        out = self._VerifyNet.forward(blobs=[score_name, ], data=data)
        return out[score_name].copy()

    def align_image(self, image, face_landmarks, crop_size=(178, 218)):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        src = np.array(
            [[face_landmarks[0, 0], face_landmarks[0, 1]],
                [face_landmarks[1, 0], face_landmarks[1, 1]],
                [(face_landmarks[4, 0]+face_landmarks[3, 0])/2,
                    (face_landmarks[4, 1]+face_landmarks[3, 1])/2]],
            np.float32)
        matrix = cv2.getAffineTransform(src, self._mean_pose)
        image_rotate = cv2.warpAffine(image, matrix, crop_size)
        return image_rotate

    def easy_extract_features(self, image, alignments, score_name='feature'):
        face_num = len(alignments)
        face_img_list = []
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image[:, :, [2, 1, 0]]  # BGR -> RGB
        for i in range(face_num):
            face_landmarks = alignments[i]
            face_img = self.align_image(image, face_landmarks)
            face_img_list.append(face_img)
        feature = self.extract_feature_batch(face_img_list, score_name)
        return feature

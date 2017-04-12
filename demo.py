from utils.face_net import FaceNet
from utils.utils import get_cosdis
import cv2

if __name__ == '__main__':
    caffe_model_path = "./model"
    facenet = FaceNet(caffe_model_path, 0)

    rose_img = cv2.imread('rose.jpg')
    bboxes, landmarks = facenet.detect_face(rose_img)
    rose_feature = facenet.easy_extract_features(rose_img, landmarks)
    jack_img = cv2.imread('jack.jpg')
    bboxes, landmarks = facenet.detect_face(jack_img)
    jack_feature = facenet.easy_extract_features(jack_img, landmarks)

    image = cv2.imread('titanic.jpg')
    bboxes, landmarks = facenet.detect_face(image)
    feature = facenet.easy_extract_features(image, landmarks)

    assert feature.shape[0] == 2
    name = []
    for i in range(2):
        rose_dis = get_cosdis(feature[i], rose_feature[0])
        jack_dis = get_cosdis(feature[i], jack_feature[0])
        # print rose_dis, jack_dis
        if rose_dis > jack_dis:
            name.append('Rose')
        else:
            name.append('Jack')

    for i in range(len(bboxes)):
        cv2.rectangle(
            image,
            (int(bboxes[i, 0]), int(bboxes[i, 1])),
            (int(bboxes[i, 2]), int(bboxes[i, 3])),
            (0, 255, 0), 1)
        cv2.putText(
            image,
            name[i],
            (int(bboxes[i, 0]), int(bboxes[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2)
        for j in range(5):
            cv2.circle(
                image,
                (int(landmarks[i, j, 0]), int(landmarks[i, j, 1])),
                2, (0, 255, 0), 2)
    cv2.imshow('haha', image)
    cv2.waitKey()

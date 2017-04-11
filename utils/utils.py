import numpy as np


def bbreg(boundingbox, reg):
    """
    boundingbox regression
    """
    reg = reg.T
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    boundingbox[:, 0] += reg[:, 0]*w
    boundingbox[:, 1] += reg[:, 1]*h
    boundingbox[:, 2] += reg[:, 2]*w
    boundingbox[:, 3] += reg[:, 3]*h
    return boundingbox


def pad(boxesA, w, h):
    """
    padding the box if it out of the image range
    w, h: width and height of the image
    """
    boxes = boxesA.copy()

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    dy = np.maximum(0, dy-1).astype(np.int)
    dx = np.maximum(0, dx-1).astype(np.int)
    y = np.maximum(0, y-1).astype(np.int)
    x = np.maximum(0, x-1).astype(np.int)
    edy = np.maximum(0, edy-1).astype(np.int)
    edx = np.maximum(0, edx-1).astype(np.int)
    ey = np.maximum(0, ey-1).astype(np.int)
    ex = np.maximum(0, ex-1).astype(np.int)

    tmpw = tmpw.astype(np.int)
    tmph = tmph.astype(np.int)

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def rerec(bbox):
    """
    convert bboxA to square
    """
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    l = np.maximum(w, h).T
    bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
    bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5
    bbox[:, 2:4] = bbox[:, 0:2] + np.repeat([l], 2, axis=0).T
    return bbox


def nms(boxes, threshold, type):
    """
    nms algorithm
    boxes: [:,0:5]
    threshold: intersection ratio under the threshold will be thrown away
    type: 'Min' or 'Union'
    returns: indexes of the boxes picked
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    prob = boxes[:, 4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(prob.argsort())
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    """
    genreate boundingboxes from the response map of the CNN
    """
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0, :, :].T
    dy1 = reg[1, :, :].T
    dx2 = reg[2, :, :].T
    dy2 = reg[3, :, :].T
    (x, y) = np.where(map >= t)
    yy = y
    xx = x
    score = map[x, y]
    reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])
    boundingbox = np.array([yy, xx]).T
    bb1 = np.fix((stride * boundingbox + 1) / scale).T
    bb2 = np.fix((stride * boundingbox + cellsize - 1 + 1) / scale).T
    score = np.array([score])
    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)
    return boundingbox_out.T


def get_cosdis(feature1, feature2):
    return np.dot(feature1.T, feature2)/(sum(feature1**2)**0.5*sum(feature2**2)**0.5)
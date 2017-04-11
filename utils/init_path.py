import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

CAFFE_PATH = '/home/qqhuang/Software/caffe'

# Add caffe to PYTHONPATH
caffe_path = os.path.join(CAFFE_PATH, 'python')
add_path(caffe_path)

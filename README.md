# Introduction
A tool for face detection, alignment and verification

# How to Run
To run this tool, you need to:
  * Install Caffe: https://github.com/BVLC/caffe
  * Change the CAFFE_PATH in "utils/init_path.py"
  * Download models from [Dropbox](https://www.dropbox.com/sh/zdopfhld02nuc1r/AABYc6ZMDE02MThdCWa5MBALa?dl=0) or [BaiduWangPan](http://pan.baidu.com/s/1pK8a979), and put them in "model/"
Then you can run "demo.py" to see the result. If everything is right, you will see tow faces are detected and their names (Rose and Jack) are labled.

# Reference
The face detection and alignment algorithm named MTCNN is from this paper:
Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
And we also get some code from https://github.com/DuinoDu/mtcnn, which is a python version of MTCNN.

The verification algorithm is from [MMLab](http://mmlab.ie.cuhk.edu.hk/) of CUHK.

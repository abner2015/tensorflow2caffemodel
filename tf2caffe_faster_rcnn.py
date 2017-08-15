import sys, argparse
import tensorflow as tf
sys.path.append('/home/root/py-faster-rcnn/caffe-fast-rcnn/python')
sys.path.append('/home/root/py-faster-rcnn/lib')
import caffe
import numpy as np
import cv2
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
import trans_tools as trans

from tensorflow.python.training import checkpoint_utils as cp


def tensor4d_transform(tensor):
    return tensor.transpose((3, 2, 0, 1))
def tensor2d_transform(tensor):
    return tensor.transpose((1, 0))   
def tf2caffe():
    checkpoint_path = "./VGGnet_fast_rcnn_iter_70000.ckpt"
    tensorName = cp.list_variables(checkpoint_path)

    cf_prototxt = "./vgg14faster-rcnn.prototxt"
    cf_model = "./vgg16faster0814.caffemodel"
    net = caffe.Net(cf_prototxt, caffe.TRAIN)
    
    for key_value in tensorName:
        key_i = key_value[0]
        nddary_data = cp.load_variable(checkpoint_path, key_i)
        try:

            if 'data' in key_i:
                pass
            elif 'weights' in key_i:
                a = key_i.split('/')
                if (len(a) == 2):
                    key_caffe = a[0]
                if (len(a) == 3):
                    key_caffe = "rpn_conv_3x3"
                if key_caffe == 'cls_score':
                    weights = tensor2d_transform(nddary_data)  # 2dim
                if key_caffe == 'bbox_pred':
                    weights = tensor2d_transform(nddary_data)  # 2dim
                if key_caffe == 'fc7':
                    weights = tensor2d_transform(nddary_data)  # 2dim
                if key_caffe == 'fc6':
                    weights = tensor2d_transform(nddary_data)  # 2dim

                if (nddary_data.ndim == 4):
                    if key_caffe == 'rpn_cls_score':
                        a = np.squeeze(nddary_data[0][0])
                        weights = tensor2d_transform(a)  # 2dim
                    elif key_caffe == 'rpn_bbox_pred':
                        a = np.squeeze(nddary_data[0][0])
                        weights = tensor2d_transform(a)  # 2dim
                    else:
                        weights = tensor4d_transform(nddary_data)
                net.params[key_caffe][0].data.flat = weights.flat
            elif 'biases' in key_i:
                a = key_i.split('/')
                if (len(a) == 2):
                    key_caffe = a[0]
                if (len(a) == 3):
                    key_caffe = "rpn_conv_3x3"
                net.params[key_caffe][1].data.flat = nddary_data.flat
            elif 'bn_gamma' in key_i:
                a = key_i.split('/')
                if (len(a) == 3):
                    key_caffe = a[1]
                else:
                    key_caffe = a[2]
                net.params[key_caffe][0].data.flat = nddary_data.flat
            elif '_gamma' in key_i:  # for prelu
                a = key_i.split('/')
                if (len(a) == 3):
                    key_caffe = a[1]
                else:
                    key_caffe = a[2]
                assert (len(net.params[key_caffe]) == 1)
                net.params[key_caffe][0].data.flat = nddary_data.flat
            elif 'mean_rgb' in key_i:
                pass
            elif 'global' in key_i:
                pass
            else:
                sys.exit("Warning!  Unknown tf:{}".format(key_i))

        except KeyError:
            print("\nWarning!  key error tf:{}".format(key_i))

    net.save(cf_model)
    print("\n- Finished.\n")

import sys, argparse
import tensorflow as tf
import caffe
import numpy as np
import cv2
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

from tensorflow.python import pywrap_tensorflow
checkpoint_path = "./vgg_16.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()


cf_prototxt = "./VGG_ILSVRC_16_layers_deploy.prototxt"
cf_model = "./vgg16.caffemodel"
net = caffe.Net(cf_prototxt, caffe.TRAIN)

def tensor4d_transform(tensor):
    return tensor.transpose((3, 2, 0, 1))
def tensor2d_transform(tensor):
    return tensor.transpose((1, 0))

for key_i in var_to_shape_map:

    try:

        if 'data' in key_i:
            pass
        elif 'weights' in key_i:
            a = key_i.split('/')
            if(len(a)==3):
                key_caffe = a[1]
            else:
                key_caffe = a[2]
            if(reader.get_tensor(key_i).ndim == 4):
                if(key_caffe == 'fc6'):
                    weights = tensor4d_transform(reader.get_tensor(key_i).reshape([7,7,512,4096])).reshape([[7,7,512,4096][3], -1])
                elif key_caffe == 'fc7':
                    a = np.squeeze(reader.get_tensor(key_i)[0][0])
                    weights = tensor2d_transform(a)#2dim
                elif key_caffe == 'fc8':
                    a = np.squeeze(reader.get_tensor(key_i)[0][0])
                    weights = tensor2d_transform(a)#2dim
                else:
                    weights = tensor4d_transform(reader.get_tensor(key_i))
            net.params[key_caffe][0].data.flat = weights.flat
        elif 'biases' in key_i:
            a = key_i.split('/')
            if (len(a) == 3):
                key_caffe = a[1]
            else:
                key_caffe = a[2]
            net.params[key_caffe][1].data.flat = reader.get_tensor(key_i).flat
        elif 'mean_rgb' in key_i:
            pass
        elif 'global' in key_i:
            pass
        else:
            sys.exit("Warning!  Unknown tf:{}".format(key_i))
    except KeyError:
        print("\nWarning!  key error mxnet:{}".format(key_i))
net.save(cf_model)
print("\n- Finished.\n")

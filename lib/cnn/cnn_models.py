from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, stride=1, padding=0, init_scale=0.02, name="conv"),
            MaxPoolingLayer(pool_size=2, stride=2, name="pool"),
            flatten(name="flatten1"),
            fc(27, 5, init_scale=0.02, name="fc1")
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            # ConvLayer2D(input_channels=3, kernel_size=5, number_filters=16, stride=1, padding=2, init_scale=0.02, name="conv1"),
            # MaxPoolingLayer(pool_size=2, stride=2, name="pool1"),
            # ConvLayer2D(input_channels=16, kernel_size=5, number_filters=16, stride=1, padding=2, init_scale=0.02, name="conv2"),
            # MaxPoolingLayer(pool_size=2, stride=2, name="pool2"),
            # flatten(name="flatten1"),
            # fc(8*8*16, 64, init_scale=0.02, name="fc1"),
            # dropout(0.2),
            # fc(64, 20, init_scale=0.02, name="fc2")


                        
            
            ConvLayer2D(input_channels=3, kernel_size=5, number_filters=16, stride=3, padding=0, init_scale=0.02, name="conv"),
            MaxPoolingLayer(pool_size=2, stride=2, name="pool"),
            flatten(name="flatten1"),
            dropout(0.5),
            fc(5*5*16, 20, init_scale=0.02, name="fc1"),
            # dropout(0.5),
            # fc(48, 20, init_scale=0.02, name="fc2")

            ########### END ###########
        )
        
        
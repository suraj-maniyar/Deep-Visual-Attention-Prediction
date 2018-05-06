
import sys
import os

import numpy as np
import skimage.io as io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time
import numpy as np
import pickle

class DeconvNet:
    def __init__(self, use_cpu=False, checkpoint_dir='./checkpoints/'):


        self.checkpoint_dir = checkpoint_dir
        self.build(use_cpu=use_cpu)

        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config = config)
        self.session.run(tf.global_variables_initializer())
        self.mini_batch_size = 1
        self.n_iterations = 1000/self.mini_batch_size;

    def dataLoadBatch(self, steps):

        X=[]
        Y=[]

        for batch_idx in range(0,self.mini_batch_size):
            I = io.imread("data/train/images/salMap_{:05d}.jpg".format((steps*self.mini_batch_size) + batch_idx))
            I = [ [(i/1.0) for i in j ] for j in I]
            X.append(I)

            labels = io.imread("data/train/salMap/salMap_{:05d}.jpg".format((steps*self.mini_batch_size) + batch_idx))[:,:,0]
            labels = [ [(i/255.0) for i in j ] for j in labels]
            Y.append(labels)

        return X,Y


    def train(self, train_stage=1, training_steps=5, restore_session=False, learning_rate=1e-4, n_epochs=20):
        if restore_session:
            step_start = restore_session()
        else:
            step_start = 0

        self.writer = tf.summary.FileWriter(self.checkpoint_dir+"model", self.session.graph)
        print("Starting training")

        for epochs in range(0,n_epochs):
            for steps in range(0,self.n_iterations):

                print('epoch: '+str(epochs) + ' run train step: '+str(steps))
                X_train, Y_train = self.dataLoadBatch(steps)

                self.train_step.run(session=self.session, feed_dict={self.x: [X_train][0], self.y: [Y_train][0], self.rate: learning_rate})

                print(self.loss.eval(session=self.session, feed_dict={self.x: [X_train][0], self.y: [Y_train][0]} ))

                if steps % 10 == 0:

                  result = self.session.run(self.merged, feed_dict={self.x: [X_train][0], self.y: [Y_train][0]})
                  step_count  = ((epochs*self.n_iterations) + steps)

                  print("step "+ str(step_count)+ " loss "+ str(self.loss))
                  self.writer.add_summary(result, step_count)
                  self.saver.save(self.session, self.checkpoint_dir+'model', global_step=step_count)

    def build(self, use_cpu=False):
        '''
        use_cpu allows you to test or train the network even with low GPU memory
        anyway: currently there is no tensorflow CPU support for unpooling respectively
        for the tf.nn.max_pool_with_argmax metod so that GPU support is needed for training
        and prediction
        '''

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.device(device):
            self.x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
            self.y = tf.placeholder(tf.float32, shape=(None, None, None))
            expected = tf.expand_dims(self.y, -1)
            self.rate = tf.placeholder(tf.float32, shape=[])

            #CONVOLUTIONAL NETWORK 1 - ##########################################################################
            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_x= self.pad_layer(self.x, [[0, 0], [35, 35], [35, 35], [0, 0]])
            self.conv_1_1 = self.conv_layer(padded_input_x, [3, 3, 3, 64], 64, 'conv_1_1' ,padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_1_1= self.pad_layer(self.conv_1_1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_1_2 = self.conv_layer(padded_input_conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2',padding = 'VALID')

            pool_1, pool_1_argmax = self.pool_layer(conv_1_2) #Pool layer with stride 2 decreases te size by 2


            #CONVOLUTIONAL NETWORK 2 -############################################################################
            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_pool_1= self.pad_layer(pool_1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_2_1 = self.conv_layer(padded_input_pool_1, [3, 3, 64, 128], 128, 'conv_2_1',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_2_1= self.pad_layer(conv_2_1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_2_2 = self.conv_layer(padded_input_conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2',padding = 'VALID')

            pool_2, pool_2_argmax = self.pool_layer(conv_2_2)  #Pool layer with stride 2 decreases te size by 2

            #CONVOLUTIONAL NETWORK 3- ###########################################################################
            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_pool_2= self.pad_layer(pool_2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_3_1 = self.conv_layer( padded_input_pool_2, [3, 3, 128, 256], 256, 'conv_3_1',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_3_1= self.pad_layer(conv_3_1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_3_2 = self.conv_layer(padded_input_conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_3_2= self.pad_layer(conv_3_2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_3_3 = self.conv_layer(padded_input_conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3',padding = 'VALID')

            pool_3, pool_3_argmax = self.pool_layer(conv_3_3)  #Pool layer with stride 2 decreases te size by 2

            #CONVOLUTIONAL NETWORK 4- ###########################################################################
            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_pool_3= self.pad_layer(pool_3, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_4_1 = self.conv_layer(padded_input_pool_3, [3, 3, 256, 512], 512, 'conv_4_1',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_4_1= self.pad_layer(conv_4_1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_4_2 = self.conv_layer(padded_input_conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_4_2= self.pad_layer(conv_4_2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_4_3 = self.conv_layer(padded_input_conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3',padding = 'VALID')

            pool_4, pool_4_argmax = self.pool_layer(conv_4_3)#Pool layer with stride 2 decreases te size by 2

            #CONVOLUTIONAL NETWORK 5- ###########################################################################
            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_pool_4= self.pad_layer(pool_4, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_5_1 = self.conv_layer(padded_input_pool_4, [3, 3, 512, 512], 512, 'conv_5_1',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_5_1= self.pad_layer(conv_5_1, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_5_2 = self.conv_layer(padded_input_conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2',padding = 'VALID')

            #Lr multiplier of 1 for filter and 2 for bais
            padded_input_conv_5_2= self.pad_layer(conv_5_2, [[0, 0], [1, 1], [1, 1], [0, 0]])
            conv_5_3 = self.conv_layer(padded_input_conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3',padding = 'VALID')

            #UP NETWORK 1- ###########################################################################

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv5_1 = self.deconv_layer(conv_5_3,W_shape=[4,4,512,512], b_shape=512, stride=2, name='deconv5_1' )
            derelu5_1 = tf.nn.relu(deconv5_1)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv5_2 = self.deconv_layer(derelu5_1, W_shape=[4,4,256,512], b_shape=256, stride=2, name='deconv5_2')
            derelu5_2 = tf.nn.relu(deconv5_2)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv5_3 = self.deconv_layer(derelu5_2,W_shape=[4,4,128,256], b_shape=128, stride=2, name='deconv5_3')
            derelu5_3 = tf.nn.relu(deconv5_3)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv5_4 = self.deconv_layer(derelu5_3,W_shape=[4,4,64,128], b_shape=64, stride=2, name='deconv5_4')
            derelu5_4 = tf.nn.relu(deconv5_4)

            #ATTENTION 1 - PRED
            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            attention1 =self.conv_layer(derelu5_4, W_shape=[3,3,64,1], b_shape=1, name='attention', padding = 'VALID')
            self.attention1c = self.crop_layer(attention1,self.x) #Crop layer with stackoverflow code. :P

            #UP NETWORK 2- ###########################################################################

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv4_1 = self.deconv_layer(conv_4_3, W_shape=[4,4,256,512], b_shape=256, stride=2, name='deconv4_1')
            derelu4_1 = tf.nn.relu(deconv4_1)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv4_2 = self.deconv_layer(deconv4_1, W_shape=[4,4,128,256], b_shape=128, stride=2, name='deconv4_2' )
            derelu4_2 = tf.nn.relu(deconv4_2)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv4_3 = self.deconv_layer(deconv4_2, W_shape=[4,4,64,128], b_shape=64, stride=2, name='deconv4_1')
            derelu4_3 = tf.nn.relu(deconv4_3)

            #ATTENTION 2 - PRED
            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            attention2 =self.conv_layer(derelu4_3, W_shape=[3,3,64,1], b_shape=1, name='attention2',stride = 1,padding = 'VALID')
            attention2c = self.crop_layer(attention2,self.x) #Crop layer with stackoverflow code. :P

            #UP NETWORK 2- ###########################################################################

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv3_1 = self.deconv_layer(conv_3_3, W_shape=[4,4,128,256], b_shape=128, stride=2, name='deconv3_1' )
            derelu3_1 = tf.nn.relu(deconv3_1)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            deconv3_2 = self.deconv_layer(deconv3_1, W_shape=[4,4,64,128], b_shape=64, stride=2, name='deconv3_2')
            derelu3_2 = tf.nn.relu(deconv3_2)

            #ATTENTION 3 - PRED
            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            attention3 =self.conv_layer(derelu3_2, W_shape=[3,3,64,1], b_shape=1, name='attention3',stride = 1,padding = 'VALID')
            attention3c = self.crop_layer(attention3,self.x) #Crop layer with stackoverflow code. :P

            ### Concat and multiscale weight layer - ALONG NUM CHANNEL DIMENSION as per prototext###
            attention = tf.concat([self.attention1c, attention2c, attention3c], axis=3)

            #Lr multiplier of 1,decay multipler is 1 for filter and Lr multiplier of 2, decay multipler is 0 for bais
            padded_attention = self.pad_layer(attention, [[0, 0], [1, 1], [1, 1], [0, 0]])
            final_attention = self.conv_layer(padded_attention, W_shape=[3,3,3,1], b_shape=1, name='final_attention',padding = 'VALID')

            ### Loss

            self.logits_1 = tf.reshape(self.attention1c, (-1, 1))
            logits_2 = tf.reshape(attention2c, (-1, 1))
            logits_3 = tf.reshape(attention3c, (-1, 1))
            logits_attention = tf.reshape(final_attention, (-1, 1))

            self.labels = tf.reshape(expected, (-1,1))

            self.loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits_1))
            self.loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits_2))
            self.loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits_3))
            self.loss_attention = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits_attention))


            with tf.name_scope('loss'):
                self.loss = (self.loss_1+self.loss_2+self.loss_3)/3.0

                tf.summary.scalar('loss1', self.loss_1)
                tf.summary.scalar('loss2', self.loss_2)
                tf.summary.scalar('loss3', self.loss_3)
                tf.summary.scalar('loss_attention', self.loss)
                tf.summary.scalar('loss_combined', self.loss)

            with tf.name_scope('train'):
                self.train_step = tf.train.GradientDescentOptimizer(self.rate).minimize(self.loss)

            self.merged = tf.summary.merge_all()

    def pad_layer(self,layer,pad_parameters):
        return tf.pad(layer, pad_parameters, "CONSTANT")

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv_layer(self, x, W_shape, b_shape, name, padding,stride = 1):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding) + b)

    def pool_layer(self, x):
          '''
          see description of build method
          '''
          with tf.device('/gpu:0'):
              return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def deconv_layer(self, x, W_shape, b_shape, name, stride=2, padding='VALID'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1]*stride, x_shape[2]*stride, W_shape[2]])

        return tf.layers.conv2d_transpose(x,W_shape[2], W_shape[0], strides=2)

    def crop_layer(self, x1,x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)

        return x1_crop

deconvNet = DeconvNet()
deconvNet.train()

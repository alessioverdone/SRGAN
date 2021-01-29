import glob
import tensorflow as tf
import numpy as np
import os
import imageio
from scipy.misc import imread, imresize
import time
from PIL import Image

tf.reset_default_graph()

device_name_gpu = "/gpu:0"
device_name_cpu = "/cpu:0"



class SRGAN(object):
    
    
    def __init__(self, sess, config, image_size=64,hr_size=256, batch_size=1,checkpoint_dir='',  np_file_dir = ''):
        self.sess = sess
        self.image_size = image_size
        self.hr_size = hr_size
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.np_file_dir = np_file_dir
        self.config = config
        #Set residual blocks
        self.n_blocks = 16
        self.graph_created = False
        self.build_final_model()

    def convlayers(self,imgs_input=None,reuse = False ):
        self.parameters = []
        
            
        with tf.variable_scope('cl'):
            if (reuse):
                tf.get_variable_scope().reuse_variables()
            # zero-mean input
            with tf.variable_scope('preprocess') :
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                # images = self.imgs-mean
                images = imgs_input-mean
    
            # conv1_1
            with tf.variable_scope('conv1_1') :
                kernel = tf.get_variable('weights', initializer=tf.compat.v1.truncated_normal([3, 3, 3, 64], 
                                        dtype=tf.float32,stddev=1e-1), trainable = False)
                                                          
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases',initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=False)
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name='cl/conv1_1')
                self.parameters += [kernel, biases]
    
            # conv1_2
            with tf.variable_scope('conv1_2') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[64], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name='cl/conv1_2')
                self.parameters += [kernel, biases]
    
            # pool1
            self.pool1 = tf.nn.max_pool(self.conv1_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool1')
    
            # conv2_1
            with tf.variable_scope('conv2_1') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name='cl/conv2_1')
                self.parameters += [kernel, biases]
    
            # conv2_2
            with tf.variable_scope('conv2_2') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name='cl/conv2_2')
                self.parameters += [kernel, biases]
    
            # pool2
            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')
    
            # conv3_1
            with tf.variable_scope('conv3_1') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name='cl/conv3_1')
                self.parameters += [kernel, biases]
    
            # conv3_2
            with tf.variable_scope('conv3_2') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name='cl/conv3_2')
                self.parameters += [kernel, biases]
    
            # conv3_3
            with tf.variable_scope('conv3_3') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[256], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name='cl/conv3_3')
                self.parameters += [kernel, biases]
    
            # pool3
            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')
    
            # conv4_1
            with tf.variable_scope('conv4_1') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name='cl/conv4_1')
                self.parameters += [kernel, biases]
    
            # conv4_2
            with tf.variable_scope('conv4_2') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name='cl/conv4_2')
                self.parameters += [kernel, biases]
    
            # conv4_3
            with tf.variable_scope('conv4_3') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name='cl/conv4_3')
                self.parameters += [kernel, biases]
    
            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')
    
            # conv5_1
            with tf.variable_scope('conv5_1') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name='cl/conv5_1')
                self.parameters += [kernel, biases]
    
            # conv5_2
            with tf.variable_scope('conv5_2') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name='cl/conv5_2')
                self.parameters += [kernel, biases]
    
            # conv5_3
            with tf.variable_scope('conv5_3') as scope:
                kernel = tf.get_variable(initializer=tf.compat.v1.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable = False)
                conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable(initializer=tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     trainable=False, name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name='cl/conv5_3')
                self.parameters += [kernel, biases]
    
            # pool5
            self.pool5 = tf.nn.max_pool(self.conv5_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')

    def fc_layers(self,reuse = False):
        with tf.variable_scope('fc'):
            if (reuse):
                tf.get_variable_scope().reuse_variables()
            # fc1
            with tf.variable_scope('fc1') as scope:
                shape = int(np.prod(self.pool5.get_shape()[1:]))
                fc1w = tf.get_variable(initializer=tf.compat.v1.truncated_normal([shape, 4096],
                                                             dtype=tf.float32,
                                                             stddev=1e-1), name='weights', trainable = False)
                fc1b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                     trainable=False, name='biases')
                pool5_flat = tf.reshape(self.pool5, [-1, shape])
                fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                self.fc1 = tf.nn.relu(fc1l)
                self.parameters += [fc1w, fc1b]
    
            # fc2
            with tf.variable_scope('fc2') as scope:
                fc2w = tf.get_variable(initializer=tf.compat.v1.truncated_normal([4096, 4096],
                                                             dtype=tf.float32,
                                                             stddev=1e-1), name='weights', trainable = False)
                fc2b = tf.get_variable(initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                     trainable=False, name='biases')
                fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                self.fc2 = tf.nn.relu(fc2l)
                self.parameters += [fc2w, fc2b]
    
            # fc3
            with tf.variable_scope('fc3') as scope:
                fc3w = tf.get_variable(initializer=tf.compat.v1.truncated_normal([4096, 1000],
                                                             dtype=tf.float32,
                                                             stddev=1e-1), name='weights', trainable = False)
                fc3b = tf.get_variable(initializer=tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                     trainable=False, name='biases')
                self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
                self.parameters += [fc3w, fc3b]

    def vgg_network(self, reuse = False, imgs_input=None):
        
        with tf.variable_scope('Vgg_network', reuse = reuse):
            self.convlayers(imgs_input, reuse)
            self.fc_layers(reuse)
            probs = tf.nn.softmax(self.fc3l)
        
        return probs
    
    def load_weights(self, weight_file, sess):
        
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))
    


    
    def Generator(self, generator_input = None, reuse = False, is_train=True):
        
        
        with tf.variable_scope('generator') as scope:   
            if (reuse):
                tf.get_variable_scope().reuse_variables()
            
            self.res_weights_dict = {}
            self.res_bias_dict = {}
            for i in range(self.n_blocks):
                self.res_weights_dict['conv0w_'+str(i)] = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 64], stddev=1e-3), name=('gen_conv0w_' + str(i)))
                self.res_bias_dict['conv0b_'+str(i)]   = tf.get_variable(initializer=tf.zeros([64]), name=('gen_conv0b_'+str(i)))
                self.res_weights_dict['conv1w_'+str(i)] = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 64], stddev=1e-3), name=('gen_conv1w_' + str(i)))
                self.res_bias_dict['conv1b_'+str(i)]   = tf.get_variable(initializer=tf.zeros([64]), name=('gen_conv1b_'+str(i)))
                
                

            self.init_weights = tf.get_variable(initializer=tf.random_normal([9, 9, 3, 64], stddev=1e-3), name=('gen_init_conv'))
            self.init_bias    = tf.get_variable(initializer=tf.zeros([64]), name=('gen_init_bias'))
            self.post_w0      = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 64], stddev=1e-3), name=('gen_post_w0'))
            self.post_b0      = tf.get_variable(initializer=tf.zeros([64]), name=('gen_post_b0'))
            self.post_w1      = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 256], stddev=1e-3), name=('gen_post_w1'))
            self.post_b1      = tf.get_variable(initializer=tf.zeros([256]), name=('gen_post_b1'))
            self.post_w2      = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 256], stddev=1e-3), name=('gen_post_w2'))
            self.post_b2      = tf.get_variable(initializer=tf.zeros([256]), name=('gen_post_b2'))
            self.post_w3      = tf.get_variable(initializer=tf.random_normal([9, 9, 64, 3], stddev=1e-3), name=('gen_post_w3'))
            self.post_b3      = tf.get_variable(initializer=tf.zeros([3]), name=('gen_post_b3'))
            
            
            
            if generator_input == None:
                generator_input = self.input_gz
            
            #Set initial weights
            
            conv_init = tf.nn.relu(tf.nn.conv2d(generator_input,self.init_weights,strides=[1,1,1,1], padding='SAME') + self.init_bias)   
            x_temp = conv_init
    
            for n in range(self.n_blocks):
                w0 = self.res_weights_dict['conv0w_'+str(n)]
                w1 = self.res_weights_dict['conv1w_'+str(n)]
                b0 = self.res_bias_dict['conv0b_'+str(n)]
                b1 = self.res_bias_dict['conv1b_'+str(n)]
                
                res1 = tf.nn.conv2d(x_temp, w0, strides=[1,1,1,1], padding='SAME') + b0 
                res1 = tf.contrib.layers.batch_norm(inputs = res1, center=True, scale=True, is_training=is_train, scope="gen_res_1_" + str(n))
                res2 = tf.nn.relu(res1)
                res3 = tf.nn.conv2d(res2, w1, strides=[1,1,1,1], padding='SAME') + b1
                res3 = tf.contrib.layers.batch_norm(inputs = res3, center=True, scale=True, is_training=is_train, scope="gen_res_2_" + str(n))
                res4 = tf.math.add(res3,x_temp)      

                x_temp = res4
                

            conv0 = tf.nn.conv2d(x_temp, self.post_w0, strides=[1,1,1,1], padding='SAME') + self.post_b0
            conv0 =  tf.contrib.layers.batch_norm(inputs = conv0, center=True, scale=True, is_training=is_train, scope="gen_1")
            conv0 = tf.math.add(conv_init,conv0)
            conv1 = tf.nn.conv2d(conv0, self.post_w1, strides=[1,1,1,1], padding='SAME') + self.post_b1  
            up1 = tf.nn.relu(tf.nn.depth_to_space(conv1,block_size=2))
            conv2 = tf.nn.conv2d(up1, self.post_w2, strides=[1,1,1,1], padding='SAME') + self.post_b2   
            up2 = tf.nn.relu(tf.nn.depth_to_space(conv2,block_size=2))
            conv3 = tf.nn.tanh(tf.nn.conv2d(up2, self.post_w3, strides=[1,1,1,1], padding='SAME') + self.post_b3)   

        return conv3

        
    def Discriminator(self,discriminator_input,reuse = False):


        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                print('-----------------------------ok')
                tf.get_variable_scope().reuse_variables()
            
            self.init_weights_dis = tf.get_variable( initializer=tf.random_normal([3, 3, 3, 64], stddev=1e-3), name=('dis_init_conv'))
            self.init_bias_dis    = tf.get_variable(initializer=tf.zeros([64]), name=('dis_init_bias'))
            #Conv1 dis
            self.dis_w1           = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 64], stddev=1e-3), name=('dis_w1'))
            self.dis_bias1        = tf.get_variable(initializer=tf.zeros([64]), name=('dis_bias1'))
            #Conv2 dis
            self.dis_w2           = tf.get_variable(initializer=tf.random_normal([3, 3, 64, 128], stddev=1e-3), name=('dis_w2'))
            self.dis_bias2        = tf.get_variable(initializer=tf.zeros([128]), name=('dis_bias2'))
            #Conv3 dis
            self.dis_w3           = tf.get_variable(initializer=tf.random_normal([3, 3, 128, 128], stddev=1e-3), name=('dis_w3'))
            self.dis_bias3        = tf.get_variable(initializer=tf.zeros([128]), name=('dis_bias3'))
            #Conv4 dis
            self.dis_w4           = tf.get_variable(initializer=tf.random_normal([3, 3, 128, 256], stddev=1e-3), name=('dis_w4'))
            self.dis_bias4        = tf.get_variable(initializer=tf.zeros([256]), name=('dis_bias4'))
            #Conv5 dis
            self.dis_w5           = tf.get_variable(initializer=tf.random_normal([3, 3, 256, 256], stddev=1e-3), name=('dis_w5'))
            self.dis_bias5        = tf.get_variable(initializer=tf.zeros([256]), name=('dis_bias5'))
            #Conv6 dis
            self.dis_w6           = tf.get_variable(initializer=tf.random_normal([3, 3, 256, 512], stddev=1e-3), name=('dis_w6'))
            self.dis_bias6        = tf.get_variable(initializer=tf.zeros([512]), name=('dis_bias6'))
            #Conv7 dis
            self.dis_w7           = tf.get_variable(initializer=tf.random_normal([3, 3, 512, 512], stddev=1e-3), name=('dis_w7'))
            self.dis_bias7        = tf.get_variable(initializer=tf.zeros([512]), name=('dis_bias7'))
            #Dense layer
            self.dense_dis_w      = tf.get_variable(initializer=tf.random_normal([16*16*512,1024], stddev=1e-3), name=('dis_w_dense'))
            self.dense_dis_b      = tf.get_variable(initializer=tf.zeros([1024]), name=('dis_b_dense'))
            self.dense_dis_w_end  = tf.get_variable(initializer=tf.random_normal([1024, 1], stddev=1e-3), name=('dis_w_dense_end'))
            self.dense_dis_b_end  = tf.get_variable(initializer=tf.zeros([1]), name=('dis_b_dense_end'))
            
            alpha = 0.2     
            momentum = 0.8
            dis0 = tf.nn.leaky_relu(tf.nn.conv2d(discriminator_input, self.init_weights_dis, strides=[1,1,1,1], padding='SAME') + self.init_bias_dis)
            dis1 = tf.nn.leaky_relu(tf.nn.conv2d(dis0, self.dis_w1, strides=[1, 2, 2, 1], padding='SAME')+ self.dis_bias1,alpha)
            dis1 = tf.contrib.layers.batch_norm(inputs = dis1, center=True, scale=True, is_training=True, scope="dis_1")
            dis2 = tf.nn.leaky_relu(tf.nn.conv2d(dis1, self.dis_w2, strides=[1, 1, 1, 1], padding='SAME')+ self.dis_bias2,alpha)
            dis2 = tf.contrib.layers.batch_norm(inputs = dis2, center=True, scale=True, is_training=True, scope="dis_2")
            dis3 = tf.nn.leaky_relu(tf.nn.conv2d(dis2, self.dis_w3, strides=[1, 2, 2, 1], padding='SAME')+ self.dis_bias3,alpha)
            dis3 = tf.contrib.layers.batch_norm(inputs = dis3, center=True, scale=True, is_training=True, scope="dis_3")
            dis4 = tf.nn.leaky_relu(tf.nn.conv2d(dis3, self.dis_w4, strides=[1, 1, 1, 1], padding='SAME')+ self.dis_bias4,alpha)
            dis4 = tf.contrib.layers.batch_norm(inputs = dis4, center=True, scale=True, is_training=True, scope="dis_4")
            dis5 = tf.nn.leaky_relu(tf.nn.conv2d(dis4, self.dis_w5, strides=[1, 2, 2, 1], padding='SAME')+ self.dis_bias5,alpha)
            dis5 = tf.contrib.layers.batch_norm(inputs = dis5, center=True, scale=True, is_training=True, scope="dis_5")
            dis6 = tf.nn.leaky_relu(tf.nn.conv2d(dis5, self.dis_w6, strides=[1, 1, 1, 1], padding='SAME')+ self.dis_bias6,alpha)
            dis6 = tf.contrib.layers.batch_norm(inputs = dis6, center=True, scale=True, is_training=True, scope="dis_6")
            dis7 = tf.nn.leaky_relu(tf.nn.conv2d(dis6, self.dis_w7, strides=[1, 2, 2, 1], padding='SAME')+ self.dis_bias7,alpha)
            dis7 = tf.contrib.layers.batch_norm(inputs = dis7, center=True, scale=True, is_training=True, scope="dis_7")
            
            dis7_flat = tf.reshape(dis7,[-1,16*16*512])#flatten
            dense_0 = tf.nn.leaky_relu( tf.matmul(dis7_flat,self.dense_dis_w) + self.dense_dis_b, alpha=0.2)
            logits_dis = tf.matmul(dense_0, self.dense_dis_w_end) + self.dense_dis_b_end
            dense_1 = tf.nn.sigmoid(logits_dis)
             

            return dense_1, logits_dis




    
    def build_final_model(self):
           
        self.imgs = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Input_Vgg_true')
        self.input_dx= tf.placeholder(tf.float32, shape = [None,256,256,3], name = 'Input_Dx')
        self.input_gz= tf.placeholder(tf.float32, shape = [None,64,64,3], name = 'Input_Gz')

        self.Dx, logits_dx = self.Discriminator(self.input_dx)
        self.Gz = self.Generator()
        self.Dg, logits_dg = self.Discriminator(self.Gz, reuse = True)      

        #Convert images in [0,255] format in order to be readable from Vgg
        cont_t = interval_mapping_tf(self.imgs, -1, 1, 0, 255)
        cont_f = interval_mapping_tf(self.Gz, -1, 1, 0, 255)
        self.content_true = self.vgg_network(imgs_input = tf.image.crop_and_resize(cont_t,boxes = [[0,0,1,1]],crop_size=[224,224], box_ind=[0]))#[0]
        self.content_false = self.vgg_network(imgs_input = tf.image.crop_and_resize(cont_f,boxes = [[0,0,1,1]],crop_size=[224,224], box_ind=[0]), reuse = True)
        
        
        
        with tf.name_scope('discriminator_loss'):
            
            d_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_dx), logits_dx))
            d_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_dg), logits_dg))
            
            self.d_loss = d_loss_real + d_loss_fake
            
        with tf.name_scope('generator_loss'):
            
            #Generator loss
            
            self.adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_dg, labels = tf.ones_like(logits_dg))) 
            self.content_loss = tf.reduce_mean(tf.square(self.content_true - self.content_false))
            self.g_loss =  1e-3 * self.adv_loss +  self.content_loss 
            
            #Pre-train network using only mse
            
            img_gen = tf.squeeze(self.Gz)#3-d channel 
            self.mse = tf.reduce_mean(tf.square(tf.squeeze(self.target) - img_gen))
            self.total_loss = self.mse 
            
            #Usint both vgg and mse
            
            # self.adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_dg, labels = tf.ones_like(logits_dg))) 
            # self.content_loss = tf.reduce_mean(tf.square(self.content_true - self.content_false))
            # img_gen = tf.squeeze(self.Gz)#3-d channel 
            # self.mse = tf.reduce_mean(10e2*tf.square(tf.squeeze(self.target) - img_gen))
            # self.g_loss =  1e-3 * self.adv_loss +  self.content_loss + self.mse
            
            
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        

        # Stochastic gradient descent with the standard backpropagation
        adam = tf.train.AdamOptimizer(learning_rate=0.0002, beta1 = 0.5)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'generator')     
        with tf.control_dependencies(update_ops):
            self.train_optimizator_gen = adam.minimize(self.g_loss, var_list = g_vars )

        update_ops_2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'discriminator')  
        with tf.control_dependencies(update_ops_2):
            self.train_optimizator_dis = adam.minimize(self.d_loss, var_list = d_vars)
 
        self.g_loss_summary = tf.summary.scalar(name = 'G_loss', tensor = self.g_loss )
        self.adv_loss_summary = tf.summary.scalar(name = 'Adv_loss', tensor = self.adv_loss )
        self.cont_loss_summary = tf.summary.scalar(name = 'Cont_loss', tensor = self.content_loss )
        self.d_loss_summary = tf.summary.scalar(name = 'D_loss', tensor = self.d_loss )
        

 
    
    def sample_images(self,data_dir, batch_size, high_resolution_shape, low_resolution_shape,all_images):
        # Make a list of all images inside the data directory
        #all_images = glob.glob(data_dir)
    
        # Choose a random batch of images
        images_batch = np.random.choice(all_images, size=batch_size)
    
        low_resolution_images = []
        high_resolution_images = []
    
        for img in images_batch:
            # Get an ndarray of the current image
            img1 = imageio.imread(img, pilmode='RGB') 
            img1 = img1.astype(np.float32)
    
            # Resize the image
            img1_high_resolution = imresize(img1, high_resolution_shape)
            img1_low_resolution = imresize(img1, low_resolution_shape)
    
            # Do a random horizontal flip
            if np.random.random() < 0.5:
                img1_high_resolution = np.fliplr(img1_high_resolution)
                img1_low_resolution = np.fliplr(img1_low_resolution)
    
            high_resolution_images.append(img1_high_resolution)
            low_resolution_images.append(img1_low_resolution)

        # Convert the lists to Numpy NDArrays
        return np.array(high_resolution_images), np.array(low_resolution_images)



    
    def save(self, checkpoint_dir, step):
        model_name = "SRGAN.model"
        model_dir = "%s_%s" % ("srgan_v1",1.1)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        number_of_version = 1.1
        model_dir = "%s_%s" % ("srgan_v1",number_of_version)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    

    def train(self, config):
        
        
        data_dir = "data2/*.*"
        epochs = config.epoch#20000
        batch_size = config.batch_size#1

        low_resolution_shape = (64, 64, 3)
        high_resolution_shape = (256, 256, 3)
        tf.global_variables_initializer().run()
       
        # Add Tensorboard
        log_dir = "logs_v1_4/".format(time.time())
        writer = tf.summary.FileWriter(log_dir, graph = tf.get_default_graph())
        #Load pretrained vgg weights
        weights = 'vgg16_weights.npz'
        self.load_weights(weights,self.sess)
        self.saver = tf.compat.v1.train.Saver()
        

        
        if self.load(self.checkpoint_dir):
          print(" [*] Load SUCCESS")
        else:
              print(" [!] Load failed...")
        
          
        d_loss = 0
        g_loss = 0
        d_loss_100=0
        g_loss_100=0
        time_100=0
        lista_d_loss = []
        lista_g_loss = []
        last_epoch = 0
        
        try:    
            np.load('np_file_4/last_epoch_file.npy')
            last_epoch = np.asscalar(np.load('np_file_4/last_epoch_file.npy'))
        except:
            last_epoch = 0
            
        all_images = glob.glob(data_dir)
        pretrain_time = 5000
        device = device_name_cpu

        with tf.device(device):
            
            print('------------------------Start training--------------------------')
            for epoch in range(last_epoch,last_epoch+epochs):

                step0 = time.time()
                # Sample a batch of images
                high_resolution_images, low_resolution_images = self.sample_images(data_dir, self.batch_size, high_resolution_shape,   low_resolution_shape,all_images)                                                              
                high_resolution_images = high_resolution_images / 127.5 - 1.
                low_resolution_images = low_resolution_images / 127.5 - 1.
           
                # Train the discriminator network on real and fake images       
                if epoch >= pretrain_time:
                    _, d_loss, d_sum = self.sess.run([self.train_optimizator_dis, self.d_loss, self.d_loss_summary], feed_dict={self.input_dx: high_resolution_images, self.input_gz: low_resolution_images})

                # Train generator network 
                high_resolution_images, low_resolution_images = self.sample_images(data_dir=data_dir, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape,all_images=all_images)                                                 
                high_resolution_images = high_resolution_images / 127.5 - 1.
                low_resolution_images = low_resolution_images / 127.5 - 1.
                
                if epoch >= pretrain_time:
                    _, g_loss, g_sum, adv_sum, cont_sum = self.sess.run([self.train_optimizator_gen, self.g_loss, self.g_loss_summary, self.adv_loss_summary, self.cont_loss_summary],
                                                                                 feed_dict={self.input_gz : low_resolution_images, self.imgs : high_resolution_images})
                
                else:        
                     _, loss= self.sess.run([self.train_optimizator_gen_prev, self.total_loss], feed_dict={self.input_gz : low_resolution_images, self.target : high_resolution_images })

                #Results each 100 iterations
                tot_time = time.time() - step0
                d_loss_100 +=d_loss
                g_loss_100 +=g_loss
                time_100 += tot_time
                if epoch%100 == 0:
                    d_loss_100 = d_loss_100/100
                    g_loss_100 = g_loss_100/100
                    time_100_2 = time_100/100
                    print("Epoch: "+str(epoch)+ "     g_loss: "+str(g_loss_100)+"     d_loss: "+str(d_loss_100) + "   Epoch_time: " + str(time_100_2))   
                    d_loss_100=0
                    g_loss_100=0
                    time_100=0


                #Summary section
                lista_d_loss.append(d_loss)
                lista_g_loss.append(g_loss)
                if epoch >= pretrain_time:
                    writer.add_summary(g_sum,epoch)
                    writer.add_summary(adv_sum,epoch)
                    writer.add_summary(cont_sum,epoch)
                    writer.add_summary(d_sum,epoch)

                #Save model and trend file each 3000 iterations
                if (epoch % 3000 == 0 and epoch != last_epoch): 
                    self.save(config.checkpoint_dir, epoch)
                    print('Model saved at epoch:  '+ str(epoch))

                   
                    epoch_arr = np.array(epoch)
                    np.save('np_file_4/last_epoch_file.npy',epoch_arr  )
                   
                    try:
                        np.load('np_file_4/list_d_loss_file.npy')#se sono già presenti i file
                        #Load arrays
                        file_d_loss = np.load('np_file_4/list_d_loss_file.npy')
                        file_g_loss = np.load('np_file_4/list_g_loss_file.npy')

                        #Transform list in arrays
                        array_d_loss = np.array(lista_d_loss)
                        array_g_loss = np.array(lista_g_loss)

                        #Concatenate arrays
                        file_d_loss = np.append(file_d_loss, array_d_loss, axis = 0)
                        file_g_loss = np.append(file_g_loss, array_g_loss, axis = 0)

                        #Save arrays
                        np.save('np_file_4/list_d_loss_file.npy', file_d_loss)
                        np.save('np_file_4/list_g_loss_file.npy', file_g_loss)

                        lista_d_loss = []
                        lista_g_loss = []

                    except:
                        #Transform list in arrays
                        array_d_loss = np.array(lista_d_loss)
                        array_g_loss = np.array(lista_g_loss)

                        #Save arrays
                        np.save('np_file_4/list_d_loss_file.npy', array_d_loss)
                        np.save('np_file_4/list_g_loss_file.npy', array_g_loss)

                        lista_d_loss = []
                        lista_g_loss = []

                       
                       
  
                if epoch % 250 == 0 : # Reproduce a reconstructed image 

                    generated_high_resolution_images = self.sess.run([self.Generator(reuse=True, is_train= False)], feed_dict={self.input_gz : low_resolution_images})
                    generated_high_resolution_images = np.array(generated_high_resolution_images[0])
                    high_resolution_images = np.array(high_resolution_images[0])
                    generated_high_resolution_images = interval_mapping(generated_high_resolution_images, -1, 1, 0, 255).astype('uint8')
                    high_resolution_images = interval_mapping(high_resolution_images, -1, 1, 0, 255).astype('uint8')
                   
                    generated_img = Image.fromarray(np.squeeze(generated_high_resolution_images, axis = 0), mode='RGB')
                    original_img = Image.fromarray(high_resolution_images, mode='RGB')
                    name_generated = "generated_images5\generated_at_epoch_" + str(epoch) + '.png'
                    orig_img_name = "generated_images5\original_at_epoch_"+str(epoch) + '.png'
                    generated_img.save(name_generated)
                    original_img.save(orig_img_name)
                
        
    
    
#These 2 functions are used to transform images from [-1,1] interval to classic [0,255]    

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

    
def interval_mapping_tf(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = (image - from_min) / float(from_range)
    return to_min + (scaled * to_range)
    




    
    

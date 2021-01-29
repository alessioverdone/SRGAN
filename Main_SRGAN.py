from Model_SRGAN import SRGAN
import tensorflow as tf
import os


flags = tf.compat.v1.app.flags
flags.DEFINE_integer("epoch", 20000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [33]")
flags.DEFINE_integer("hr_size", 256, "The size of label to produce [21]")
flags.DEFINE_string("checkpoint_dir", "Check_dir_v1_4", "Name of checkpoint directory [checkpoint3]")
flags.DEFINE_string("log_directory", "logs_v1_4", "Name of checkpoint directory [checkpoint3]")
flags.DEFINE_string("np_file_dir", "np_file_4", "Name of sample directory [sample3]")
FLAGS = flags.FLAGS


def main(_):

  if not os.path.exists(FLAGS.checkpoint_dir):#Se non esiste il path,crealo
    os.makedirs(FLAGS.checkpoint_dir)
  print('Start program...')

  with tf.Session() as sess:
    srgan = SRGAN(sess, 
                  FLAGS,
                  image_size=FLAGS.image_size, 
                  hr_size=FLAGS.hr_size, 
                  batch_size=FLAGS.batch_size, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  np_file_dir=FLAGS.np_file_dir)

    srgan.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()

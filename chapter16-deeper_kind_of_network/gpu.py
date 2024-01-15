import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.test.gpu_device_name()
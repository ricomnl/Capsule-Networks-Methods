from sklearn.cross_validation import train_test_split
from plant_seedings import model
import numpy as np
import tensorflow as tf

data = np.load('./out.npz')
X_train, X_val, y_train, y_val = train_test_split(data['arr_0'], data['arr_1'], test_size = 0.2)

caps1_n_maps = 16
caps1_n_caps = caps1_n_maps * 27 * 27 
caps1_n_dims = 16

caps2_n_caps = 12
caps2_n_dims = 32

n_hidden1 = 400
n_hidden2 = 1024

parameters =  { "caps1_n_maps": caps1_n_maps,
                "caps1_n_caps": caps1_n_caps,
                "caps1_n_dims": caps1_n_dims,
                "kernel_prime": 5,
                "caps2_n_caps": caps2_n_caps,
                "caps2_n_dims": caps2_n_dims,
                "n_hidden1": n_hidden1,
                "n_hidden2": n_hidden2 }

model(X_train, y_train, X_val, y_val, parameters, batch_size = 50, img_size=(32,32,3), restore_checkpoint=True)

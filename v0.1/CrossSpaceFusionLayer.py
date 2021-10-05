'''
Created on Jul 24, 2019

Fusion layers for Keras v2.
Will learn weights that will help correlate information through the last dimension of the
input matrix and will output a dimensionally reduced matrix -> the last dimension will
be lost

@author: cmihaigabriel
@version: 1.0 

@todo: ensure H and W as needed or adapt to other input_shapes
@todo: implement unit_center option
'''

from keras import backend as K
from keras.layers import Layer
from keras.initializers import Initializer
from keras.layers import multiply, add
from keras.backend import squeeze, eval
import numpy as np
from scipy.constants.constants import alpha

'''
Will create a Fusion Layer that will transform from 3-dims to 2-dims matrix.
A SIMPLE example of a Fusion3to2 using a 3X3X2 input matrix would look like this:
------------------------
Layer operations:
  Input last dim 1     || Input last dim 2     => Output
  
   _______W_________
 | X8 --- X1 --- X2     || C8 --- C1 --- C2        (a8 * R + b8 * R * C8)/2 --- (a1 * R + b1 * R * C1)/2 --- (a2 * R + b2 * R * C2)/2     
H| X7 --- R  --- X3     || C7 --- CI --- C3     => (a7 * R + b7 * R * C7)/2 --- R                        --- (a3 * R + b3 * R * C3)/2
 | X6 --- X5 --- X4     || C6 --- C5 --- C4        (a6 * R + b6 * R * C6)/2 --- (a5 * R + b5 * R * C5)/2 --- (a4 * R + b4 * R * C4)/2
------------------------
Will learn {a1...a8} and {b1...b8}
It will force the middle output value to always be R - equal to the initial 
input value - this applies well to the conditions of the initial experiment  
------------------------

@attention: this layer will always organize in a square pattern - W must be a 
multiple of H -> otherwise it will not work or it will crash? 
'''

class CSF3var (Layer):
    
    def __init__ (self, force_unitcenter, **kwargs):
        self.force_unitcenter = force_unitcenter
        super(CSF3var, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.walpha = self.add_weight(name = 'alphas', 
                                      shape = (input_shape[1], input_shape[2]),
                                      initializer='random_uniform',
                                      trainable=True)
        self.wbeta = self.add_weight(name = 'betas',
                                      shape = (input_shape[1], input_shape[2]),
                                      initializer='random_uniform',
                                      trainable=True)
        self.inputsh = input_shape
        
        super(CSF3var, self).build(input_shape)
        
    def call(self, x):
        x_scores = x[:,:,:,1]
        x_coefs = x[:,:,:,2]
        x_centroids = x[:,:,:,0]
        w_beta = self.wbeta[None, :, :]
        w_alpha = self.walpha[None, :, :]
        
        beta_runs = multiply([x_scores, x_coefs])
        beta_component = multiply([beta_runs, w_beta])
        
        alpha_component = multiply([x_centroids, w_alpha])
        
        ret_matrix = add([alpha_component, beta_component])
        
        return ret_matrix
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])
 

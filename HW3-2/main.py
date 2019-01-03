import tensorflow as tf
import numpy as np
from model import DCGAN
import os

epoch = 20
batch_size = 64
noise_dim = 128
g_n = 2
d_n = 3
model_dir = './model/'
train = 1 # 0 for test
save = 'model'

model = DCGAN(noise_dim)
if train == 1:
	model.train(epoch, batch_size, g_n, d_n, model_dir, save)
if train == 0:
	model.test(model_dir)
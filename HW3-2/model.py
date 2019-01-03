import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import os
import cv2

class DataSet():
	def __init__(self):
		self.image_num = 36739
		self.image_path = './data/extra_data/images/'
		self.img_npy_path = './data/extra_data/faces.npy'
		self.tag_path = './data/extra_data/tags.csv'
		self.image_list = []
		self.load_data()

	def load_data(self):		
		if os.path.isfile(self.img_npy_path):
			self.image_list = np.load(self.img_npy_path)
		else:
			image_list = []
			for i in tqdm(range(self.image_num)):
				image_bgr = cv2.resize(cv2.imread(self.image_path+str(i)+".jpg"), (64, 64))
				image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
				image_list.append(image_rgb)
			image_list = np.array(image_list)
			self.image_list = image_list / 128 - 1
			np.save(self.img_npy_path, self.image_list)
		print('Image loaded')
		
		tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair',	'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
		tag_csv = open(self.tag_path, 'r').readlines()
		
		label_list = []
		for i in range(len(tag_csv)):
			id, tag = tag_csv[i].split(',')
			label = np.zeros(len(tag_dict),dtype=float)
			for j in range(len(tag_dict)):
				if tag_dict[j] in tag:
					label[j] = 1.
			label_list.append(label)
		self.label_list = np.array(label_list)

	def get_batch(self, batch_size, batch_needed):
		index = np.arange(self.image_num)
		np.random.shuffle(index)
		images = np.array([ self.image_list[idx] for idx in index ])
		labels = np.array([ self.label_list[idx] for idx in index ])
		wrong_labels = labels.copy()
		np.random.shuffle(wrong_labels)
		
		batch_num = self.image_num // batch_size
		batched_images = np.split(images[: batch_size*batch_num ], batch_num)
		batched_labels = np.split(labels[: batch_size*batch_num ], batch_num)
		batched_wrong_labels = np.split(wrong_labels[: batch_size*batch_num ], batch_num)

		return batched_images, batched_labels, batched_wrong_labels


class DCGAN():
	def __init__(self, noise_dim):
		self.pic_path = './examples/'
		if not os.path.exists(self.pic_path):
			os.makedirs(self.pic_path)
		self.noise_dim = noise_dim
		self.alpha = 0.2
		self.learning_rate = 0.0002
		self.beta1 = 0.5
		self.dropout_rate = 0.2
		self.display_step = 100
		
		self.test_labels = [np.zeros(23) for _ in range(25)]
		for i in range( 0, 5):
			self.test_labels[i][8] = 1
			self.test_labels[i][22] = 1
		for i in range( 5,10):
			self.test_labels[i][8] = 1
			self.test_labels[i][19] = 1
		for i in range(10,15):
			self.test_labels[i][8] = 1
			self.test_labels[i][21] = 1
		for i in range(15,20):
			self.test_labels[i][4] = 1
			self.test_labels[i][22] = 1
		for i in range(20,25):
			self.test_labels[i][4] = 1
			self.test_labels[i][21] = 1

		plt.switch_backend('agg')
		self.build_model()
		self.saver = tf.train.Saver(max_to_keep=50)

	def discriminator(self, inputs, condition, reuse=False, is_training=True):
		with tf.variable_scope('discriminator', reuse=reuse):
			conv1 = tf.nn.leaky_relu((tf.layers.conv2d(
                inputs=inputs, filters=128, kernel_size=(5,5), strides=(2,2), padding="SAME",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))), alpha=self.alpha)
			conv2 = tf.nn.leaky_relu((tf.layers.conv2d(
                inputs=conv1, filters=256, kernel_size=(5,5), strides=(2,2), padding="SAME",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))), alpha=self.alpha)
			conv3 = tf.nn.leaky_relu((tf.layers.conv2d(
                inputs=conv2, filters=512, kernel_size=(5,5), strides=(2,2), padding="SAME",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))), alpha=self.alpha)
			embed_y = tf.expand_dims(condition, 1)
			embed_y = tf.expand_dims(embed_y, 2)
			tiled_embeddings = tf.tile(embed_y, [1, 8, 8, 1])
			concat = tf.concat([conv3, tiled_embeddings], -1)
			conv4 = tf.nn.leaky_relu((tf.layers.conv2d(
                inputs=concat, filters=512, kernel_size=(1,1), strides=(1,1), padding="SAME",
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))), alpha=self.alpha)
			flatten = tf.layers.flatten(conv4)
			output = tf.layers.dense(flatten,1)
		return output

	def generator(self, inputs, condition, reuse=False, is_training=True):
		with tf.variable_scope('generator', reuse=reuse):
			inputs = tf.concat([inputs, condition], -1)
			dense1 = tf.nn.leaky_relu(tf.layers.dense(inputs, 1024*8*8))
			reshape = tf.reshape(dense1, shape=[-1, 8, 8, 1024])
			deconv1 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(
                reshape, filters=512, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
			deconv2 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(
                deconv1, filters=256, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
			deconv3 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(
                deconv2, filters=128, kernel_size=(4,4), strides=(2,2), padding='same'), alpha=self.alpha)
			deconv4 = tf.nn.leaky_relu(tf.layers.conv2d_transpose(
                deconv3, filters=3, kernel_size=(4,4), strides=(1,1), padding='same'), alpha=self.alpha)
			output = tf.nn.tanh(deconv4)
		return output

	def build_model(self):
		self.g_inputs = tf.placeholder(shape=(None, self.noise_dim), dtype=tf.float32, name='generator_inputs')
		self.d_inputs = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32, name="discriminator_inputs")
		self.label = tf.placeholder(shape=(None, 23), dtype=tf.float32, name='train_label')
		self.wrong_label = tf.placeholder(shape=(None, 23), dtype=tf.float32, name='train_wrong_label')

		self.g_sample = self.generator(self.g_inputs, self.label, tf.AUTO_REUSE, is_training=True)
		self.d_real = self.discriminator(self.d_inputs, self.label, tf.AUTO_REUSE, is_training=True)
		self.d_fake = self.discriminator(self.g_sample, self.label, tf.AUTO_REUSE, is_training=True)
		self.d_wrong_label = self.discriminator(self.d_inputs, self.wrong_label, tf.AUTO_REUSE, is_training=True)
		
		self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_real, labels=tf.ones_like(self.d_real)))
		self.d_fake_loss = ( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits=self.d_fake, labels=tf.zeros_like(self.d_fake))) 
                +tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_wrong_label, labels=tf.zeros_like(self.d_wrong_label))) ) / 2
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.d_fake, labels=tf.ones_like(self.d_fake)))
		self.d_loss = self.d_real_loss + self.d_fake_loss
		self.g_infer = self.generator(self.g_inputs, self.label, tf.AUTO_REUSE, is_training=False)
		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
		self.d_optim = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1).minimize( self.d_loss, var_list=d_vars )
		self.g_optim = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1).minimize( self.g_loss, var_list=g_vars )

	def get_noise(self, batch_size):
 		return np.clip(np.random.normal(0, 0.4, (batch_size, self.noise_dim)), -1, 1)
		
	def train(self, epochs, batch_size, g_iter, d_iter, model_dir, model_name):
		print('Training . . .')
		dp = DataSet()
		sample_noise = self.get_noise(5*5)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print ('Reloading model parameters...')
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			
			for epoch in range(epochs):
				d_loss, g_loss = 0,0
				batched_img, batched_label, batched_wrong_label = dp.get_batch(batch_size, d_iter)
				dataset = list(zip(batched_img, batched_label, batched_wrong_label))
				
				for b, (img, label, wrong_label) in enumerate(dataset):
					z = self.get_noise(batch_size)
					for it in range(d_iter):
						_, d_loss, d_real, d_fake = sess.run(
                            [ self.d_optim, self.d_loss, self.d_real_loss, self.d_fake_loss ],
                            feed_dict={ self.g_inputs:z, self.d_inputs:img, self.label:label,
                                       self.wrong_label:wrong_label })
					z = self.get_noise(batch_size)
					for it in range(g_iter):
						_, g_loss = sess.run(
                            [ self.g_optim, self.g_loss ], feed_dict={ self.g_inputs:z, self.label:label })
					print('Epoch: {:>2}/{:>2} / [{:>4}/{:>4}] / D_loss: {:.6f} / G_loss: {:.6f}'
						  .format(epoch+1, epochs, b+1, len(batched_img), d_loss, g_loss) )
					if (b+1) % self.display_step == 0:
						samples = sess.run(self.g_infer, feed_dict={ 
                            self.g_inputs:sample_noise, self.label:self.test_labels })
						samples = samples / 2 + 0.5
						fig = self.visualize_result(samples)
						plt.savefig(self.pic_path+'{}.png'.format(str((epoch+1)*10000+b+1)), bbox_inches='tight')
						plt.close(fig)
			if model_name != '':
				self.saver.save(sess, './model/{}'.format(model_name))
				print('Model saved at \'{}\''.format('./model'))
			
	def test(self, model_dir):
		sample_noise = self.get_noise(5*5)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			ckpt = tf.train.get_checkpoint_state(model_dir)
			print(ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				self.saver.restore(sess, ckpt.model_checkpoint_path)
			samples = sess.run(self.g_infer, feed_dict={ 
                self.g_inputs:sample_noise, self.label:self.test_labels })
			samples = samples / 2 + 0.5
			fig = self.visualize_result(samples)
			plt.savefig('./test.png', bbox_inches='tight')
			plt.close(fig)
			
		
	def visualize_result(self, samples):
		fig = plt.figure(figsize=(5,5))
		gs = gridspec.GridSpec(5,5)
		gs.update(wspace=0.1, hspace=0.1)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig


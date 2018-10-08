import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''
TF_CPP_MIN_LOG_LEVEL
0: default, displaying all logs
1: filter out INFO logs
2: filter out INFO + WARNINGS
3: filter out INFO + WARNINGS + ERROR logs
'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)


'''
Computation Graph
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784]) # [height: any, width: flattened 28 x 28]
y = tf.placeholder('float')

def nn_model(data):

	hl1 = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hl2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hl3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	outl = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
			  'biases': tf.Variable(tf.random_normal([n_classes]))}

	# input * weight + bias
	# biases could make the weights still updating even if the inputs are all 0

	l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, outl['weights']) + outl['biases']

	return output




def nn_train(x):
	prediction = nn_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	# default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

n_epochs = 10

'''
Session
'''
# 1st way
with tf.Session() as sess:
	sess.run(tf.tf.global_variables_initializer())

	for epoch in range(n_epochs):
		epoch_cost = 0

		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
			epoch_cost += c
		print('Epoch', epoch, 'completed out of', n_epochs, 'with cost', epoch_cost)

	correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))



nn_train(x)





'''
# 2nd way
sess = tf.Session()
# do sth
sess.close()
'''


	

import tensorflow as tf

# Variable in tensorflow

string = tf.Variable("My name is Yogesh",tf.string);
number = tf.Variable(45,tf.int16)
floating = tf.Variable(3.567,tf.float64) 

# Rank in tensorflow

rank1_tensor = tf.Variable(['test'],tf.string)
rank2_tensor = tf.Variable([['ok','ok','ok'],['Yogesh','Yogesh','Yogesh']])
tf.rank(rank2_tensor)

# Shape of tensor

rank2_tensor.shape
TensorShape([2, 3])

# Reshaping the Tensor

demo_tensor = tf.ones([1,2,3])
demo_tensor

reshaped1_tensor = tf.reshape(demo_tensor,[2,3,1])
reshaped1_tensor

reshaped2_tensor = tf.reshape(demo_tensor,[3,-1])
reshaped2_tensor

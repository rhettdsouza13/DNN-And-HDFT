import tensorflow as tf

x = tf.Variable([-10.5432])

y = tf.nn.relu(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print sess.run(y.eval())

import tensorflow as tf

# 两个矩阵的对应元素各自相乘！！
x = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
y = tf.constant([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
# 注意这里这里x,y要有相同的数据类型，不然就会因为数据类型不匹配而出错
z = tf.multiply(x, y)

# 两个数相乘
x1 = tf.constant(1)
y1 = tf.constant(2)
# 注意这里这里x1,y1要有相同的数据类型，不然就会因为数据类型不匹配而出错
z1 = tf.multiply(x1, y1)

# 数和矩阵相乘
x2 = tf.constant([[1.0, 2.0, 3.0], [5.0, 2.0, 7.0], [1.0, 2.0, 3.0]])
y2 = tf.constant([2.0, 3.0, 1.0])
a2 = tf.constant([2.0, 3.0, 1.0])
b2 = tf.constant([2.0, 2.0, 2.0])
# 注意这里这里x1,y1要有相同的数据类型，不然就会因为数据类型不匹配而出错
z2 = tf.multiply(x2, y2)
z21 = tf.add(x2, y2)
z22 = tf.truediv(x2, b2)
z23 = tf.reciprocal(x2)

# 两个矩阵相乘
x3 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
y3 = tf.constant([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
# 注意这里这里x,y要满足矩阵相乘的格式要求。
z3 = tf.matmul(x, y)

with tf.Session() as sess:
    # print(sess.run(z))
    # print(sess.run(z1))
    print(sess.run(z2))
    print(sess.run(z21))
    print(sess.run(z23))
    # print(sess.run(z3))

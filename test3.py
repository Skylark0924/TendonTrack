import tensorflow as tf


def test_kal():
    ipk_basic_action = [tf.ones(4) * 2, tf.ones(4) * 2]
    mu = tf.constant([-0.1196, -0.04713,  0.1777 ,  0.00102])
    log_sigma = tf.constant([0.16789, -0.22875,  0.40902, -0.55863])

    mu, log_sigma = tf.multiply(ipk_basic_action, tf.add(tf.ones(4), mu)), tf.add(
        0.5 * tf.log(tf.square(ipk_basic_action)), log_sigma)
    return mu, log_sigma


if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(test_kal()))

import numpy as np
import tensorflow as tf


class SOM:

    def __init__(self, n_features, lattice, time_steps, lr=0.5, sigma=None):

        sigma = sigma if sigma is not None else np.max(lattice)/2

        self.n_features = n_features
        self.lattice_shape = lattice
        self.time_steps = time_steps

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[n_features])

        # tf variables
        self.lattice = tf.Variable(np.random.random((np.prod(lattice), n_features)), dtype=tf.float32)
        self.t = tf.Variable(0, dtype=tf.float32)

        # tf constants
        self.lr_0 = tf.constant(lr, dtype=tf.float32)
        self.sigma_0 = tf.constant(sigma, dtype=tf.float32)
        self.lr_lambda = tf.constant(time_steps, dtype=tf.float32)
        self.n_lambda = tf.constant(time_steps/np.log(sigma), dtype=tf.float32)
        self.neighbourhood_matrix = tf.constant(-self.generate_neighbourhood_matrix(lattice)**2, dtype=tf.float32)

        # ops
        self.lr = self.lr_0 * tf.exp(tf.negative(tf.divide(self.t, self.lr_lambda)))
        self.sigma = self.sigma_0 * tf.exp(tf.negative(tf.divide(self.t, self.n_lambda)))
        self.input_node_diff = tf.subtract(self.inputs, self.lattice)
        self.bmu = tf.argmin(tf.reduce_sum(tf.abs(self.input_node_diff), 1))
        self.neighbourhood_factor = tf.exp(self.neighbourhood_matrix[self.bmu]/(2*tf.square(self.sigma)))
        self.update_factor = self.lr*self.neighbourhood_factor
        self.update_w = self.lattice.assign_add(tf.expand_dims(self.update_factor, 1)*self.input_node_diff)
        self.increment_t = self.t.assign_add(1)

        self.session = None

    def get_session(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        return self.session

    def do_iteration(self, input_vectors):
        np.random.shuffle(input_vectors)
        for input_vector in input_vectors:

            self.get_session().run(self.update_w, feed_dict={self.inputs: input_vector})
        self.get_session().run(self.increment_t)

    def get_weights(self):
        return self.get_session().run(tf.reshape(self.lattice, (np.concatenate((self.lattice_shape, [self.n_features])))))

    @staticmethod
    def generate_neighbourhood_matrix(lattice_shape):
        n_nodes = np.prod(lattice_shape)
        ret = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for i in range(n_nodes):
            for j in range(n_nodes):
                i_coord = np.unravel_index(i, lattice_shape)
                j_coord = np.unravel_index(j, lattice_shape)
                ret[i, j] = np.sqrt(np.sum(np.square(np.subtract(i_coord, j_coord))))

        return ret


if __name__ == '__main__':

    data = np.array([[0, 0], [100, 100], [0, 100], [100, 0]])
    net = SOM(2, [20], 200, lr=0.1)

    for _ in range(2000):
        net.do_iteration(data)

    print(net.get_weights())
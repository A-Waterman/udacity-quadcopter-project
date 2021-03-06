from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, sess, state_size, action_size, action_low, action_high, learning_rate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.learning_rate = learning_rate

        self.sess = sess
        K.set_session(sess)

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                   name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        """
        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        q_values = layers.Input(shape=(1,))

        # loss = -K.mean(action_gradients * actions)
        loss = -K.mean(action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
             inputs=[self.model.input, action_gradients, K.learning_phase()],
             outputs=[self.model.output],
             updates=updates_op)
        #inputs=[self.model.input, action_gradients, K.learning_phase()],
        """
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        self.params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradient)
        zipped_grads = zip(self.params_grad, self.model.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zipped_grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_gradients):
        self.sess.run(self.optimize,
                      feed_dict={
                          self.model.input: states,
                          self.action_gradient: action_gradients
                      })

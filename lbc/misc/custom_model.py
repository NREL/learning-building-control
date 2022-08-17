from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf1, tf, tfv = try_import_tf()


OUTPUT_LEN = 6


def get_customized_model(input_len,
                         network_shape,
                         hidden_layer_activation=tf.tanh,
                         last_layer_activation=None):
    """ Generate a customized policy network model using the given parameters.

    Args:
      input_len: int, the dimension of the observation vector, which varies by
        DR type.
      network_shape: list of int, indicating the neuron numbers of hidden 
        layers.
      hidden_layer_activation: tf activation function class, indicating what
        activation function to be used in the hidden layers. Default to be
        tanh function.
      last_layer_activation: tf activation function class, indicating what
        activation function to be used in the last layer. Default to be None, 
        meaning no activation needed. 

    """

    class CustomFullyConnectedNetwork(TFModelV2):

        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name):

            super(CustomFullyConnectedNetwork, self).__init__(
                obs_space, action_space, num_outputs, model_config, name)

            inputs = tf.keras.layers.Input(shape=(input_len,), name="obs1")

            # 1. Policy network
            next_layer = inputs

            for idx in range(len(network_shape)):
                next_layer = tf.keras.layers.Dense(
                    network_shape[idx], name='fc_' + str(idx+1),
                    activation=hidden_layer_activation)(next_layer)

            output_mean = tf.keras.layers.Dense(
                OUTPUT_LEN, name='out_mean',
                activation=last_layer_activation)(next_layer)

            output_variance = tf.keras.layers.Dense(
                OUTPUT_LEN, name='out_variance', activation=None)(next_layer)

            layer_out = tf.concat((output_mean, output_variance), 1)

            # 2. Value network
            next_layer = inputs

            for idx in range(len(network_shape)):
                next_layer = tf.keras.layers.Dense(
                    network_shape[idx], name='fc_value_' + str(idx + 1),
                    activation=hidden_layer_activation)(next_layer)

            value_out = tf.keras.layers.Dense(
                1, name='fc_value_out', activation=None)(next_layer)

            self.base_model = tf.keras.Model(inputs, [layer_out, value_out])
            self.register_variables(self.base_model.variables)

        def forward(self, input_dict, state, seq_lens):
            model_out, self._value_out = self.base_model(input_dict["obs"])
            return model_out, state

        def value_function(self):
            return tf.reshape(self._value_out, [-1])

    class CustomModel(TFModelV2):
        """Example of a custom model that just delegates to a fc-net."""

        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name):

            super(CustomModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name)

            self.model = CustomFullyConnectedNetwork(
                obs_space, action_space, num_outputs, model_config, name)
            self.register_variables(self.model.variables())

        def forward(self, input_dict, state, seq_lens):
            return self.model.forward(input_dict, state, seq_lens)

        def value_function(self):
            return self.model.value_function()

    return CustomModel

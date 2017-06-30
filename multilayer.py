class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        if not isinstance(layer, Layer): raise TypeError('The layer is not a valid object!', layer)
        if self.layers:
            layer.build(self.layers[len(self.layers)-1].hidden_units)
            self.layers.append(layer)
        else: self.layers.append(layer.build())


class Layer: # object parameter is no longer needed in python 3
    def __init__(self, **kwargs):
        self.built = False

        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'batch_size',
                          'dtype',
                          'name',
                          'trainable',
                          'weights',
                          'input_dtype',  # legacy
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # implement kwargs, etc.

class Linear(Layer):
    def __init__(self, hidden_units, activation = None, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.input_shape = kwargs.get('input_shape')
        self.weights = None

        # add other methods later, like data regularization,

    def build(self, input_shape):
        self.input_shape = input_shape






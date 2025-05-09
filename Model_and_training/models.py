import torch
import torch.nn as nn
import numpy as np

class FFNN(torch.nn.Module):
    def __init__(self, input_size, layers, activation = None, last_activation = nn.Sigmoid(), last_number_of_nodes = 3, dropout = 0, batch_norm = False, last_bias = True):
        super(FFNN, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm

        if type(layers) != list:
            raise ValueError('layers must be a list of integers')
        
        if not isinstance(layers, list):
            raise ValueError('layers must be a list of integers')

        if activation is None:
            activation = [nn.ReLU() for _ in range(len(layers))]  # Default to ELU for each layer
        elif not isinstance(activation, list):
            activation = [type(activation)() for _ in range(len(layers))]  # Create new instances for each layer


        if dropout != 0:
            setattr(self, 'dropout', nn.Dropout(p=dropout))

        setattr(self, 'fcin', nn.Linear(input_size, layers[0]))

        for i in range(0,len(layers)-1):
            setattr(self, 'activation'+str(i), activation[i])
            setattr(self, 'fc'+str(i), nn.Linear(layers[i], layers[i+1]))
            if dropout != 0:
                setattr(self, 'dropout'+str(i), nn.Dropout(p=dropout))
            if batch_norm:
                setattr(self, 'batch_norm'+str(i), nn.BatchNorm1d(layers[i+1]))
        
        setattr(self, 'activation'+str(len(layers)-1), activation[-1])
        setattr(self, 'fc'+str(len(layers)-1), nn.Linear(layers[-1], last_number_of_nodes, bias = last_bias))

        self.n_layers = len(layers)
        self.activation = activation
        self.last_activation = last_activation
        self.layers = layers

    def init_weights(self, pure_zero = False):
        for layer in [self.fcin] + [getattr(self, f'fc{i}') for i in range(self.n_layers)]:
            if pure_zero:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            else:   
                if isinstance(layer, nn.Linear):
                    if self.activation.__class__.__name__ == 'ReLU':
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    elif self.activation.__class__.__name__ == 'LeakyReLU':
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                    else:  # ELU or other activations
                        nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        return self


        
        
    def forward(self, x):

        x = getattr(self, 'fcin')(x)
        if self.dropout != 0:
            x = getattr(self, 'dropout')(x)

        for i in range(self.n_layers-1):
            x = getattr(self, 'activation'+str(i))(x)
            x = getattr(self, 'fc'+str(i))(x)
            if self.dropout != 0:
                x = getattr(self, 'dropout'+str(i))(x)
            if self.batch_norm:
                x = getattr(self, 'batch_norm'+str(i))(x)

        x = getattr(self, 'activation'+str(self.n_layers-1))(x)
        x = getattr(self, 'fc'+str(self.n_layers-1))(x)
        if self.last_activation is not None:
            x = self.last_activation(x)

        return x



class CNN(torch.nn.Module):

    def check_input(self, conv_dims, pool_sizes,linear_layers, 
                        pool_functions = None, conv_activation = None, 
                        linear_activation = None, last_linear_activation = None, 
                        list_of_cnn_args = None):
        
        if type(conv_dims) != np.ndarray:
            print(conv_dims)
            raise ValueError('conv_dims must be a numpy array')
        if conv_dims.ndim != 2:
            print(conv_dims)
            raise ValueError('conv_dims must be a 2D numpy array')

        if conv_dims.shape[1] != 3:
            print(conv_dims)
            raise ValueError('Each row of conv_dims must have 3 elements')

        if len(conv_dims) != len(pool_sizes):
            raise ValueError('conv_dims and pool_sizes must have the same length')

        if pool_functions is None:
            pool_functions = [nn.MaxPool2d for _ in range(len(conv_dims))]

        if not isinstance(pool_functions, list):
            pool_functions = [type(pool_functions) for _ in range(len(conv_dims))]

        if conv_activation is None:
            conv_activation = [nn.ReLU() for _ in range(len(conv_dims))]

        if linear_activation is None:
            # Default to ReLU for each layer
            linear_activation = [nn.ReLU() for _ in range(len(linear_layers))]  
        elif not isinstance(linear_activation, list):
            linear_activation = [type(linear_activation)() for _ in range(len(
                linear_activation))]  # Create new instances for each layer
    
        if list_of_cnn_args is None:
            list_of_cnn_args = [{} for _ in range(len(conv_dims))]


        return pool_functions, conv_activation, linear_activation, \
                last_linear_activation, list_of_cnn_args

    def __init__(self, input_shape, conv_dims, pool_sizes,linear_layers, 
                        pool_functions = None, conv_activation = None, 
                        linear_activation = None, last_linear_activation = None, 
                        list_of_cnn_args = None):
        super(CNN, self).__init__()

        pool_functions, conv_activation, linear_activation, last_linear_activation,\
        list_of_cnn_args = self.check_input(
            conv_dims, pool_sizes, linear_layers, pool_functions, 
            conv_activation, linear_activation, last_linear_activation, 
            list_of_cnn_args)
        
        self.conv_dims = conv_dims
        self.n_linear_layers = len(linear_layers)
        self.linear_activation = linear_activation
        self.pool_sizes = pool_sizes
        self.list_of_cnn_args = list_of_cnn_args
        self.last_linear_activation = last_linear_activation
        print(self.list_of_cnn_args)
        
        for i, conv_layer in enumerate(conv_dims):
            setattr(self, 'conv'+str(i), nn.Conv2d(*conv_layer, **list_of_cnn_args[i]))
            setattr(self, 'conv_activation'+str(i), conv_activation[i])
            if pool_sizes[i] is not None:
                setattr(self, 'pool'+str(i), pool_functions[i](pool_sizes[i], stride = 2))

        linear_input = self.find_linear_layer_size(input_shape, conv_dims, pool_sizes)
        setattr(self, 'fcin', nn.Linear(linear_input, linear_layers[0]))

        for i in range(0,len(linear_layers)-1):
            setattr(self, 'linear_activation'+str(i), linear_activation[i])
            setattr(self, 'fc'+str(i), nn.Linear(linear_layers[i], linear_layers[i+1]))
        
        setattr(self, 'linear_activation'+str(len(linear_layers)-1), linear_activation[-1])
        setattr(self, 'fc'+str(len(linear_layers)-1), nn.Linear(linear_layers[-1], 1))


    def calc_stagnante_padding(self, kernel_size):
        return (kernel_size - 1) // 2

    def calc_output_size(self, input_size, kernel_size, padding, stride=1):
        return (input_size - kernel_size + 2*padding) // stride + 1
    
    def find_linear_layer_size(self, input_size, conv_layers, pool_layers):

        output_sizes = [[*input_size]] 

        for i, conv_layer in enumerate(conv_layers):
            temp_output = [conv_layer[1]]
            if "padding" in self.list_of_cnn_args[i]:
                padding = self.list_of_cnn_args[i]["padding"]
            else:
                padding = 0
            temp_output.append(self.calc_output_size(output_sizes[-1][1], conv_layer[2], 
                                                padding))
            temp_output.append(self.calc_output_size(output_sizes[-1][2], conv_layer[2],
                                                padding))
            output_sizes.append(temp_output)
            if pool_layers[i] is not None:
                temp_output = [conv_layers[i][1]]
                temp_output.append(self.calc_output_size(output_sizes[-1][1], pool_layers[i][0], 0,
                                                    2))
                temp_output.append(self.calc_output_size(output_sizes[-1][2], pool_layers[i][1], 0,
                                                    2))
                output_sizes.append(temp_output)    
        output_sizes = np.array(output_sizes)
        print(output_sizes)
        return np.prod(output_sizes[-1])
        
    def forward(self, x):

        for i in range(len(self.conv_dims)):
            x = getattr(self, 'conv'+str(i))(x)
            x = getattr(self, 'conv_activation'+str(i))(x)
            if self.pool_sizes[i] is not None:
                x = getattr(self, 'pool'+str(i))(x)

        x = x.flatten(start_dim=1)
        x = getattr(self, 'fcin')(x)


        for i in range(self.n_linear_layers-1):
            x = getattr(self, 'linear_activation'+str(i))(x)
            x = getattr(self, 'fc'+str(i))(x)

        x = getattr(self, 'linear_activation'+str(self.n_linear_layers-1))(x)
        x = getattr(self, 'fc'+str(self.n_linear_layers-1))(x)
        if self.last_linear_activation is not None:
            x = self.last_linear_activation(x)
        return x

    def init_weights(self):
        for i in range(len(self.conv_dims)):
            torch.nn.init.xavier_uniform_(getattr(self, 'conv'+str(i)).weight)
            torch.nn.init.zeros_(getattr(self, 'conv'+str(i)).bias)

        torch.nn.init.xavier_uniform_(getattr(self, 'fcin').weight)
        torch.nn.init.zeros_(getattr(self, 'fcin').bias)

        for i in range(self.n_linear_layers):
            torch.nn.init.xavier_uniform_(getattr(self, 'fc'+str(i)).weight)
            torch.nn.init.zeros_(getattr(self, 'fc'+str(i)).bias)
            
        return self

import tensorflow as tf
from .gcn import ConvTemporalGraphical
from .graph import Graph

class Model(object):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels,
                num_class,
                edge_importance_weighting=True, 
                **kwargs):
        
        # Load graph
        self.graph = Graph(layout='customer settings',
                            strategy='spatial',
                            max_hop=1,
                            dilation=1)
        self.A = tf.constant(self.graph.A, dtype='float32', name='A')

        spatial_kernel_size = self.A.get_shape()[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # Some necessary layers
        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)
        self.fcn = tf.keras.layers.Conv2D(num_class, kernel_size=(1,1), data_format='channels_first')

        # Built st_gcn networks
        self.st_gcn1 = st_gcn(in_channels, 64, kernel_size, stride=1, dropout=0)
        self.st_gcn2 = st_gcn(64, 64, kernel_size, stride=1, dropout=0)
        self.st_gcn3 = st_gcn(64, 64, kernel_size, stride=1, dropout=0)
        self.st_gcn4 = st_gcn(64, 64, kernel_size, stride=1, dropout=0)
        self.st_gcn5 = st_gcn(64, 128, kernel_size, stride=2, dropout=0)
        self.st_gcn6 = st_gcn(128, 128, kernel_size, stride=1, dropout=0)
        self.st_gcn7 = st_gcn(128, 128, kernel_size, stride=1, dropout=0)
        self.st_gcn8 = st_gcn(128, 256, kernel_size, stride=2, dropout=0)
        self.st_gcn9 = st_gcn(256, 256, kernel_size, stride=1, dropout=0)
        self.st_gcn10 = st_gcn(256, 256, kernel_size, stride=1, dropout=0)

        # Initialize Mask weight - parameters for edge importance weighting
        # WARNING: IF YOU CHANGE NUMBER OF ST_GCN LAYER, YOU MUST CHANGE num_stgcn VALUE BELLOW
        k, v, w = self.A.get_shape()
        num_stgcn = 10
        if edge_importance_weighting == True:
            self.edge_importance = []
            for i in range(num_stgcn):
                mask = tf.Variable(tf.ones_like(self.A))
                self.edge_importance.append(mask)
                # self.edge_importance = tf.Variable(self.edge_importance)
        else:
            self.edge_importance = tf.ones([num_stgcn, k, v, w])
            # self.edge_importance = tf.constant(self.edge_importance)

        

    def forward(self, x):
        # Normalize data
        N, C, T, V, M = x.get_shape()
        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
        x = tf.reshape(x, [N*M, V*C, -1], name=None)
        x = self.data_bn(x)
        x = tf.reshape(x, [N, M, V, C, -1], name=None)
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        x = tf.reshape(x, [N*M, C, -1, V], name=None)

        # Forward through st_gcn networks
        x, _ = self.st_gcn1.forward(x, self.A*self.edge_importance[0])
        x, _ = self.st_gcn2.forward(x, self.A*self.edge_importance[1])
        x, _ = self.st_gcn3.forward(x, self.A*self.edge_importance[2])
        x, _ = self.st_gcn4.forward(x, self.A*self.edge_importance[3])
        x, _ = self.st_gcn5.forward(x, self.A*self.edge_importance[4])
        x, _ = self.st_gcn6.forward(x, self.A*self.edge_importance[5])
        x, _ = self.st_gcn7.forward(x, self.A*self.edge_importance[6])
        x, _ = self.st_gcn8.forward(x, self.A*self.edge_importance[7])
        x, _ = self.st_gcn9.forward(x, self.A*self.edge_importance[8])
        x, _ = self.st_gcn10.forward(x, self.A*self.edge_importance[9])

        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
        x = tf.reshape(x, [N, M, -1, 1, 1], name=None)
        x = tf.reduce_mean(x, axis=1)

        # Prediction and return logits
        x = self.fcn(x)
        x = tf.reshape(x, [x.get_shape()[0], -1], name=None)

        return x


class st_gcn(object):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dropout=0,
                residual=True):
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        # padding = ((kernel_size[0] - 1) //2 , 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                                out_channels, 
                                (kernel_size[0], 1), 
                                (stride, 1), 
                                padding='same', 
                                data_format='channels_first'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Dropout(dropout)
        ])  

        if not residual:
            self.residual = lambda x : 0
        
        elif (in_channels == out_channels) and stride ==1 :
            self.residual = lambda x : x

        else:
            self.residual = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_channels, 
                                    kernel_size=(1, 1), 
                                    strides=(stride, 1), 
                                    padding='valid',
                                    data_format='channels_first'),
                tf.keras.layers.BatchNormalization(axis=1)
            ])
        
        self.relu = tf.keras.layers.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn.forward(x, A)     # Co can chay ham forward cua gcn khong???
        x = self.tcn(x) + res

        return self.relu(x), A
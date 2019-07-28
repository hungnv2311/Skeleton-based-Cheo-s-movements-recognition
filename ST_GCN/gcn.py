import tensorflow as tf

class ConvTemporalGraphical(object):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape: Input always have CHANNEL_FIRST FORMAT
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
                 t_kernel_size=1,
                 t_stride=1,
                 t_dilation=1,
                 use_bias=True,
                 data_format='channels_first'):
        
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(out_channels * kernel_size,
                                    kernel_size=(t_kernel_size, 1),
                                    strides=(t_stride, 1),
                                    padding='valid',
                                    data_format=data_format,
                                    dilation_rate=(t_dilation, 1),
                                    use_bias=use_bias)

    def forward(self, x, A):
        assert A.get_shape()[0] == self.kernel_size
        print(x)
        x = self.conv(x)

        n, kc, t, v = x.get_shape()
        
        x = tf.reshape(x, [n, self.kernel_size, kc//self.kernel_size, -1, v], name=None)     # Use // because reshape function doesn't support / operation, it same as /  
        # Tensor contraction
        x = tf.einsum('nkctv,kvw->nctw', x, A)

        return x, A




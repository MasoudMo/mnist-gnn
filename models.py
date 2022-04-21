import torch.nn as nn
from math import floor
import torch
from torch_geometric.nn import Sequential, MessagePassing, GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool


class CNNResBlock(nn.Module):
    """
    3D convolution block with residual connections
    # Code is copied from https://github.com/RCL-LVLD/gnn_lvl

    Attributes
    ----------
    conv: torch.nn.Conv3d, PyTorch Conv3D model
    bn: torch.nn.BatchNorm3d, PyTorch 3D batch normalization layer
    pool: torch.nn.AvgPool3d, PyTorch average 3D pooling layer
    dropout: torch.nn.Dropout3D, PyTorch 3D dropout layer
    one_by_one_cnn: torch.nn.Conv3d, pyTorch 1*1 conv model to equalize the number of channels for residual addition

    Methods
    -------
    forward(x): model's forward propagation
    """
    def __init__(self,
                 in_channels: int,
                 padding: int,
                 out_channels: int = 128,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 cnn_dropout_p: float = 0.0):
        """
        :param in_channels: int, number of input channels
        :param padding: int, 0 padding dims
        :param out_channels: int, number of filters to use
        :param kernel_size: int, filter size
        :param pool_size: int, pooling kernel size for the spatial dims
        :param cnn_dropout_p: float, cnn dropout rate
        """

        super().__init__()

        # Check if a Conv would be needed to make the channel dim the same
        # for the residual
        self.one_by_one_cnn = None
        if in_channels != out_channels:
            # noinspection PyTypeChecker
            self.one_by_one_cnn = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1)

        # 2D conv layer
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(padding, padding))

        # Other operations
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
        self.dropout = nn.Dropout2d(p=cnn_dropout_p)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*1*64*T*H*W
        :return: Tensor of shape N*out_channels*T*H'*W'
        """

        # Make the number of channels equal for input and output if needed
        if self.one_by_one_cnn is not None:
            residual = self.one_by_one_cnn(x)
        else:
            residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = x + residual
        x = self.pool(x)
        x = F.relu(x)

        return self.dropout(x)


class CNN(nn.Module):
    """
    3D convolution network
    # Code is copied from https://github.com/RCL-LVLD/gnn_lvl

    Attributes
    ----------
    conv: torch.nn.Sequential, the convolutional network containing residual blocks
    output_fc: torch.nn.Sequential, the FC layer applied to the output of convolutional network

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 out_channels: list,
                 kernel_sizes: list = None,
                 pool_sizes: list = None,
                 fc_output_dim: list = None,
                 cnn_dropout_p: float = 0.0):
        """
        :param out_channels: list, output channels for each layer
        :param kernel_sizes: list, kernel sizes for each layer
        :param pool_sizes: list, pooling kernel sizes for each layer
        :param fc_output_dim: int, the output dimension of output FC layer (set to None for no output fc)
        :param cnn_dropout_p: float, dropout ratio of the CNN
        """

        super().__init__()

        n_conv_layers = len(out_channels)

        # Default list arguments
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [1]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Compute paddings to preserve temporal dim
        paddings = list()
        for kernel_size in kernel_sizes:
            paddings.append(floor((kernel_size - 1) / 2))

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(CNNResBlock(in_channels=1,
                                               padding=paddings[0],
                                               out_channels=out_channels[0],
                                               kernel_size=kernel_sizes[0],
                                               pool_size=pool_sizes[0],
                                               cnn_dropout_p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(CNNResBlock(in_channels=out_channels[layer_num-1],
                                                   padding=paddings[layer_num],
                                                   out_channels=out_channels[layer_num],
                                                   kernel_size=kernel_sizes[layer_num],
                                                   pool_size=pool_sizes[layer_num],
                                                   cnn_dropout_p=cnn_dropout_p)))
        # Change to sequential
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = None
        if fc_output_dim is not None:
            self.output_fc = nn.Sequential(nn.AdaptiveAvgPool3d((None, 1, 1)),
                                           nn.Flatten(start_dim=2),
                                           nn.Linear(out_channels[-1], fc_output_dim),
                                           nn.ReLU(inplace=True))

    def forward(self,
                x):
        """
        Forward path of the CNN3D network

        :param x: torch.tensor, input torch.tensor of image frames

        :return: Vector embeddings of input images of shape (num_samples, output_dim)
        """

        # CNN layers
        x = self.conv(x)

        # FC layer
        if self.output_fc is not None:
            x = self.output_fc(x)

        return x


class GCNClassifier(nn.Module):
    def __init__(self,
                 input_feature_dim,
                 dropout_p,
                 gnn_hidden_dims,
                 mlp_hidden_dim,
                 num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        # GNN layers
        for gnn_hidden_dim in gnn_hidden_dims:
            self.layers.append(Sequential('x, edge_index', [(GCNConv(in_channels=input_feature_dim,
                                                                     out_channels=gnn_hidden_dim), 'x, edge_index -> x'),
                                                            nn.BatchNorm1d(gnn_hidden_dim),
                                                            nn.Dropout(p=dropout_p),
                                                            nn.ReLU(inplace=True)]))
            input_feature_dim = gnn_hidden_dim

        # Output MLP layers
        self.output_mlp = nn.Sequential(nn.Linear(in_features=gnn_hidden_dims[-1],
                                                  out_features=mlp_hidden_dim),
                                        nn.BatchNorm1d(mlp_hidden_dim),
                                        nn.Dropout(p=dropout_p),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=mlp_hidden_dim,
                                                  out_features=num_classes))

    def forward(self, g):

        h = g.x
        edge_index = g.edge_index

        # GNN layers
        for gnn_layer in self.layers:
            h = gnn_layer(h, edge_index)

        # Pool node embeddings to create the graph embedding
        h = global_mean_pool(h, g.batch)

        # Output MLP
        h = self.output_mlp(h)

        return h

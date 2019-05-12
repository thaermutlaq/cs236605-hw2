import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        features_layer = [in_features] + hidden_features + [num_classes]
        for in_l, out_l in zip(features_layer, features_layer[1:]):
            blocks.append(Linear(in_l, out_l))
            if out_l != num_classes:
                blocks.append(ReLU() if activation=='relu' else Sigmoid())
                if dropout > 0.0:
                    blocks.append(Dropout(dropout))
            
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        conv_layers = [in_channels] + self.filters
        count = 0
        for c_in, c_out in zip(conv_layers, conv_layers[1:]):
            count += 1
            layers.append(nn.Conv2d(c_in, c_out, 3, padding=1))
            layers.append(nn.ReLU())
            if count == self.pool_every:
                count = 0
                layers.append(nn.MaxPool2d(2))
            
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        
        calc = int(self.filters[-1] * (in_h/2**(len(self.filters)/self.pool_every)) * (in_w/2**(len(self.filters)/self.pool_every)))
        hidden_dim = [calc] + self.hidden_dims +[self.out_classes]
        for dim_in, dim_out in zip(hidden_dim, hidden_dim[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            if dim_out != self.out_classes:
                layers.append(nn.ReLU())
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
        self.in_K = 64
        _, h_in, w_in = in_size
        hc_in = h_in
        wc_in = w_in
        for i in range(3):
            hc_in = int(((hc_in-1)/2)+1)
            wc_in = int(((wc_in-1)/2)+1)
        from itertools import groupby
        x=[[k,len(list(v))] for k,v in groupby(filters)]
        L = x[0][1]
        K = [a[0] for a in x]
        conv_start = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        layer1 = self._make_layer(K[0], L, stride=1)
        layer2 = self._make_layer(K[1] if len(K) > 1 else K[-1], L, stride=2)
        layer3 = self._make_layer(K[2] if len(K) > 2 else K[-1], L, stride=2)
        layer4 = self._make_layer(K[3] if len(K) > 3 else K[-1], L, stride=2)
        self.classifier = nn.Linear(int(int(hc_in/4)*int(wc_in/4)*K[-1]), out_classes)
        self.feature_extractor = nn.Sequential(*[conv_start, layer1, layer2, layer3, layer4])
    # ========================
    
    def _make_layer(self, K, L, stride):
        strides = [stride] + [1]*(L-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_K, K, stride))
            self.in_K = K
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class ResNetBlock(nn.Module):

    def __init__(self, in_K, K, stride=1):
        super(ResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_K)
        self.conv1 = nn.Conv2d(in_K, K, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(K)
        self.conv2 = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_K != K:
            self.shortcut = nn.Sequential(nn.Conv2d(in_K, K, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

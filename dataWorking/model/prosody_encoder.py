import torch

import torch.nn as nn
from collections import OrderedDict

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    # def __init__(self, model_config):
    def __init__(self):
        super(VariancePredictor, self).__init__()

        # self.input_size = model_config["transformer"]["encoder_hidden"]
        # self.filter_size = model_config["variance_predictor"]["filter_size"]
        # self.kernel = model_config["variance_predictor"]["kernel_size"]
        # self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        # self.dropout = model_config["variance_predictor"]["dropout"]
        self.input_size = 80
        self.filter_size = 256 
        self.kernel = 3
        self.conv_output_size = 256
        self.dropout = 0.5

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [   # (B, N, L)
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    # ("layer_norm_1", nn.functional.normalize(self.filter_size, dim=1)),
                    ("normalize", nn.BatchNorm1d(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("normalize", nn.BatchNorm1d(self.filter_size)),
                    # ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        # self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.linear_layer = nn.Conv1d(self.conv_output_size, 1, 3, 1, 1)
        

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        print(f'Out1.shape {out.shape}')
        out = self.linear_layer(out)
        print(f'Out2.shape {out.shape}')
        out = out.squeeze(-1)
        

        if mask is not None:
            out = out.masked_fill(mask.bool(), 0.0)
        print(f'Out3.shape {out.shape}')

        return out
    
class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous() # (B, N, L)
        x = self.conv(x)    # (B, out_channels, L)
        x = x.contiguous()  # (B, out_channels, L)
        print(f'Conv.shape {x.shape}')

        return x

class ProsodyEncoder(nn.Module):
    # def __init__(self, model_config):
    def __init__(self):
        super(ProsodyEncoder, self).__init__()
        # self.pitch_linear = VariancePredictor(model_config)
        # self.energy_linear = VariancePredictor(model_config)
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.n_bins = 256

        self.pitch_bins = nn.Parameter(
                torch.linspace(0, 555, self.n_bins - 1),
                requires_grad=False,
            )
        self.energy_bins = nn.Parameter(
                torch.linspace(0, 555, self.n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            256, 80
        )
        self.energy_embedding = nn.Embedding(
            256, 80
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        print(f'Prediction.shape {prediction.shape}')
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins).long())
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction.squeeze(1), self.pitch_bins)
            )
            print(f'Embedding.shape {embedding.shape}')
        return prediction, embedding.transpose(1, 2)

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction.squeeze(1), self.energy_bins)
            )
            print(f'Embedding.shape {embedding.shape}')
        return prediction, embedding.transpose(1, 2)

    def forward(
        self,
        x,
        src_mask,
        pitch_target=None,
        energy_target=None,
        p_control=1.0,
        e_control=1.0,
    ):
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_mask, p_control
        )
        x = x + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask, e_control
        )
        x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
        )


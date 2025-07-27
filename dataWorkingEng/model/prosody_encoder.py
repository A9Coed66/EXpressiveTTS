import torch

import torch.nn as nn
from collections import OrderedDict
import sys

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    # def __init__(self, model_config):
    def __init__(self):
        super(VariancePredictor, self).__init__()
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

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        # self.linear_layer = nn.Conv1d(self.conv_output_size, 1, 3, 1, 1)
        # self.activation = nn.Tanh()
        

    def forward(self, encoder_output, mask):        # encoder_output: (B, H, L), mask: (B, 1, L)
        out = self.conv_layer(encoder_output)       # (B, H, L)
        out = self.linear_layer(out.transpose(1, 2)) # (B, L, 1)
        out = out.squeeze(-1)                        # (B, L)
        
        # print(f'Out2.shape {out.shape}')
        

        if mask is not None:
            out = out * mask.squeeze(1)
        # print(f'Out3.shape {out.shape}, {out}')
        # sys.exit()

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
        # print(f'Conv.shape {x.shape}')

        return x

class ProsodyEncoder(nn.Module):
    # def __init__(self, model_config):
    def __init__(self):
        super(ProsodyEncoder, self).__init__()
        # self.pitch_linear = VariancePredictor(model_config)
        # self.energy_linear = VariancePredictor(model_config)
        # self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.n_bins = 256

        # self.pitch_bins = nn.Parameter(
        #         torch.linspace(-1, 1, self.n_bins - 1),
        #         requires_grad=False,
        #     )
        self.energy_bins = nn.Parameter(
                torch.linspace(-1, 1, self.n_bins - 1),
                requires_grad=False,
            )

        # self.pitch_embedding = nn.Embedding(
        #     256, 80
        # )
        self.energy_embedding = nn.Embedding(
            256, 80
        )

    # def get_pitch_embedding(self, x, target, mask, control):
    #     prediction = self.pitch_predictor(x, mask)
    #     # print(f'Prediction.shape {prediction.shape}')
    #     if target is not None:
    #         embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins).long())
    #     else:
    #         # print(f'Prediction before change {prediction}')
    #         prediction = prediction * control
    #         # print(f'Prediction after change {prediction}')
    #         embedding = self.pitch_embedding(
    #             torch.bucketize(prediction.squeeze(1), self.pitch_bins)
    #         )
    #         # print(f'Embedding.shape {embedding}')
    #     return prediction, embedding.transpose(1, 2)
    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction + control
            prediction = torch.clamp(prediction, min=-1.0, max=1.0)
            embedding = self.pitch_embedding(
                torch.bucketize(prediction.squeeze(1), self.pitch_bins)
            )
        if mask is not None:
            embedding = embedding.transpose(1, 2)*mask
        else:
            embedding = embedding.transpose(1, 2)
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        # print(control)
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            # print(f'Prediction before change {prediction}')
            prediction = prediction + control
            # print(f'Prediction after change {prediction}')
            prediction = torch.clamp(prediction, min=-1.0, max=1.0)
            # print(f'Prediction after change {prediction}')
            # print(f'Prediction shape {prediction.shape}')
            embedding = self.energy_embedding(
                torch.bucketize(prediction.squeeze(1), self.energy_bins)
            )   # (B, L, 80)
            # print(f'Prediction shape {prediction.shape}')

            # print(f'Embedding.shape {embedding.shape}')
        if mask is not None:
            embedding = embedding.transpose(1, 2)*mask
        else:
            embedding = embedding.transpose(1, 2)
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        pitch_target=None,
        energy_target=None,
        p_control=0.0,
        e_control=0.0,
    ):
        # print(e_control)
        # pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #     x, pitch_target, src_mask, p_control
        # )
        # print(f'Shape x: {x.shape}, pitch_embedding: {pitch_embedding.shape}')
        # x = x + pitch_embedding
        x = x.detach()
        # pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #     x, pitch_target, src_mask, p_control
        # )
        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask, e_control
        )

        return (
            # pitch_embedding,
            energy_embedding, 
            # pitch_prediction, 
            energy_prediction
        )
    
    # @torch.no_grad()
    # def forward(
    #     self,
    #     x,
    #     src_mask,
    #     pitch_target=None,
    #     energy_target=None,
    #     p_control=1.0,
    #     e_control=1.0,
    # ):
    #     pitch_prediction, pitch_embedding = self.get_pitch_embedding(
    #         x, pitch_target, src_mask, p_control
    #     )
    #     x = x + pitch_embedding
    #     energy_prediction, energy_embedding = self.get_energy_embedding(
    #         x, energy_target, src_mask, e_control
    #     )
    #     x = x + energy_embedding

    #     return (
    #         pitch_embedding,
    #         energy_embedding,
    #         pitch_prediction,
    #         energy_prediction,
    #     )

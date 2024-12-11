import math

import torch

from model.base import BaseModule
from model.utils import sequence_mask, convert_pad_shape
from model.retnet import *
from model.retnet_cfg import *
import os
import argparse
import yaml

class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, 
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels, dim=1))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels, dim=1))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

# class DurationPredictor(torch.nn.Module):
#     def __init__(self, idim, n_chans, kernel_size=3, n_layers=2, dropout_rate=0.1, offset=1.0):
#         super(DurationPredictor, self).__init__()
#         self.offset = offset
#         self.conv = torch.nn.ModuleList()
#         for idx in range(n_layers):
#             in_chans = idim if idx == 0 else n_chans
#             self.conv += [
#                 torch.nn.Sequential(
#                     torch.nn.Conv1d(in_chans,n_chans,kernel_size,stride=1,padding=(kernel_size - 1) // 2,),
#                     torch.nn.ReLU(),
#                     LayerNorm(n_chans, dim=1),
#                     torch.nn.Dropout(dropout_rate),
#                 )
#             ]
#         self.linear = torch.nn.Linear(n_chans, 1)
#     def _forward(self, xs, x_masks=None, is_inference=False):
#         # xs = xs.transpose(1, -1)  # (B, idim, Tmax)
#         for f in self.conv:
#             xs = f(xs)  # (B, C, Tmax)

#         # NOTE: calculate in log domain
#         # print(xs.shape, x_masks.shape)

#         if is_inference:
#             # NOTE: calculate in linear domain
#             xs = torch.clamp(
#                 torch.round(xs.exp() - self.offset), min=0
#             ).long()  # avoid negative value
#         if x_masks is not None:
#             xs = self.linear((xs * x_masks).transpose(1, 2)).squeeze(-1)  # (B, Tmax)
#         else:
#             xs = self.linear(xs.transpose(1, 2)).squeeze(-1)  # (B, Tmax)
#         print(xs.shape)
#         return xs

#     def forward(self, xs, x_masks=None):
#         """Calculate forward propagation.

#         Args:
#             xs (Tensor): Batch of input sequences (B, Tmax, idim).
#             x_masks (ByteTensor, optional):
#                 Batch of masks indicating padded part (B, Tmax).

#         Returns:
#             Tensor: Batch of predicted durations in log domain (B, Tmax).

#         """
#         return self._forward(xs, x_masks, False)

#     def inference(self, xs, x_masks=None):
#         """Inference duration.

#         Args:
#             xs (Tensor): Batch of input sequences (B, Tmax, idim).
#             x_masks (ByteTensor, optional):
#                 Batch of masks indicating padded part (B, Tmax).

#         Returns:
#             LongTensor: Batch of predicted durations in linear domain (B, Tmax).

#         """
#         return self._forward(xs, x_masks, True)
    
class VariancePredictor(torch.nn.Module):
    def __init__(self, idim, n_chans, kernel_size=3, n_layers=2, dropout_rate=0.1, offset=1.0):
        super(VariancePredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(in_chans,n_chans,kernel_size,stride=1,padding=(kernel_size - 1) // 2,),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)
    def _forward(self, xs, x_masks=None, is_inference=False):
        # xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        # print(xs.shape, x_masks.shape)

        if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(
                torch.round(xs.exp() - self.offset), min=0
            ).long()  # avoid negative value
        if x_masks is not None:
            xs = self.linear((xs * x_masks).transpose(1, 2)).squeeze(-1)  # (B, Tmax)
        else:
            xs = self.linear(xs.transpose(1, 2)).squeeze(-1)  # (B, Tmax)
        # print(xs.shape)
        return xs

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks, True)
    
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def pad(self, input_ele, mel_max_length=80):
        if mel_max_length:
            max_len = mel_max_length
        else:
            max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

        out_list = list()
        for i, batch in enumerate(input_ele):
            if len(batch.shape) == 1:
                one_batch_padded = F.pad(
                    batch, (0, max_len - batch.size(0)), "constant", 0.0
                )
            elif len(batch.shape) == 2:
                one_batch_padded = F.pad(
                    batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
                )
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = self.pad(output, max_len)
        else:
            output = self.pad(output)

        return output, torch.LongTensor(mel_len).to('cuda')

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class TextEncoder(nn.Module):
    # TODO: add preprocess_config to init func
    def __init__(self, n_vocab, n_feats, n_channels, filter_channels, filter_channels_dp, n_heads, n_layers, kernel_size, p_dropout, use_softmax, use_decay, preprocess_config_path='/home2/tuannd/tuanlha/EXpressiveTTS/dataWorking/preprocess.yaml', model_config_path = '/home2/tuannd/tuanlha/EXpressiveTTS/dataWorking/model.yaml', window_size=None, spk_emb_dim=64, n_spks=1):
        super(TextEncoder, self).__init__()
        self.n_vocab            = n_vocab
        self.n_feats            = n_feats
        self.n_channels         = n_channels
        self.filter_channels    = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.kernel_size = kernel_size
        self.p_dropout   = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks      = n_spks

        self.emb = torch.nn.Embedding(n_vocab, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)
        
        CONFIG = RetNetConfig(decoder_layers=n_layers,
                            decoder_embed_dim=n_channels + (spk_emb_dim if n_spks > 1 else 0),
                            decoder_value_embed_dim=n_channels + (spk_emb_dim if n_spks > 1 else 0),
                            decoder_retention_heads=n_heads,
                            decoder_ffn_embed_dim=filter_channels,
                            dropout=p_dropout,
                            )
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--model_config", type=str, default=model_config_path)
        # parser.add_argument("--preprocess_config", type=str, default=preprocess_config_path)
        # args = parser.parse_args()

        self.model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
        self.preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
        
        self.encoder = RetNetModel(CONFIG, tensor_parallel=False, use_softmax=use_softmax, use_decay=use_decay)
        self.proj_m  = torch.nn.Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1)
        self.duration_predictor  = VariancePredictor(idim=n_channels + (spk_emb_dim if n_spks > 1 else 0), n_chans=filter_channels_dp, 
                                        kernel_size=kernel_size,dropout_rate= p_dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor     = VariancePredictor(idim=n_channels + (spk_emb_dim if n_spks > 1 else 0), n_chans=filter_channels_dp, 
                                        kernel_size=kernel_size,dropout_rate= p_dropout)
        self.energy_predictor    = VariancePredictor(idim=n_channels + (spk_emb_dim if n_spks > 1 else 0), n_chans=filter_channels_dp,  
                                        kernel_size=kernel_size,dropout_rate= p_dropout)

        self.pitch_feature_level = self.preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = self.preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = self.model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = self.model_config["variance_embedding"]["energy_quantization"]
        n_bins = self.model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(self.preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, self.model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, self.model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    

    # def get_mask_from_lengths(self, lengths, max_len=None):
    #     batch_size = lengths.shape[0]
    #     if max_len is None:
    #         max_len = torch.max(lengths).item()

    #     ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to('cuda')
    #     mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    #     return mask
    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to('cuda')
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask

    def forward(self, x, x_lengths, sty, spk=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,):
        # NOTE: scale up embedding -> tránh vấn đề vanishing gradients hoặc exploding gradients
        # mu: mean ò encoded features
        # logw: logarithm of the predicted durations of the encoded features
        # x_mask: mask of the input sequence

        x = self.emb(x) * math.sqrt(self.n_channels)    # (B, T, H)
        x = torch.transpose(x, 1, -1)                   # (B, T, H) -> (B, H, T)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)    # (B, 1, H, T)

        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x  = self.encoder(inputs_embeds=x.transpose(1,2), attention_mask=x_mask, sty=sty)[0].transpose(1,2) * x_mask
        x = x * x_mask
        # mu=x
        x_dp = torch.detach(x)
        # print(x.shape)
        # print(f'x_mask: {x_mask.shape}')
        # print('Go to duration predictor')
        log_duration_prediction = self.duration_predictor(x_dp, x_mask)
        # print('Out of duration predictor')
        # print(f'log_duration_prediction.shape: {log_duration_prediction.shape}')
        # print(x.shape)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, x_mask, p_control
            )
            pitch_embedding = torch.transpose(pitch_embedding, 1, -1)
            # print(f'Phoneme level: {x.shape}, {pitch_embedding.shape}')
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, x_mask, p_control
            )
            energy_embedding = torch.transpose(energy_embedding, 1, -1)
            # print(f'Phoneme level: {x.shape}, {energy_embedding.shape}')
            x = x + energy_embedding
        # print(f'X shape af ter before embedding: {x.shape}')
        if duration_target is not None:
            # print('Duration target is not None')
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            # print('Duration target is None')
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = self.get_mask_from_lengths(mel_len)
        # print(f'X shape after length regulator: {x.shape}')
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding
        # print(f'X shape after embedding: {x.shape}')
        # duration_rounded = torch.clamp(
        #         (torch.round(torch.exp(logw) - 1) * d_control),
        #         min=0,
        #     )
        # x, mel_len = self.length_regulator(x, duration_rounded, max_len)
        # mel_mask = self.get_mask_from_lengths(mel_len)

        # if self.pitch_feature_level == "frame_level":
        #     pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #         x, pitch_target, mel_mask, p_control
        #     )
        #     x = x + pitch_embedding
        # if self.energy_feature_level == "frame_level":
        #     energy_prediction, energy_embedding = self.get_energy_embedding(
        #         x, energy_target, mel_mask, p_control
        #     )
        #     x = x + energy_embedding
        # predictions được sử dụng để so với target tính loss
        # print(f'logw.shape: {log_duration_prediction.shape}')
        # print(f'x_mask.shape: {x_mask.shape}')
        return x, log_duration_prediction, x_mask
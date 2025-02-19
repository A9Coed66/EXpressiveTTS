""" from https://github.com/jaywalnut310/glow-tts """

import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], 
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss

def pitch_loss(p_prediction, p_std_target):
    p_prediction = p_prediction[p_prediction != 0]
    p_prediction_avg = p_prediction.mean()
    p_prediction_std = p_prediction.std()
    # p_avg_loss = torch.sum((p_prediction_avg - p_avg_target)**2)/2
    p_std_loss = torch.sum((p_prediction_std - p_std_target)**2)
    return p_std_loss

def energy_loss(e_prediction, e_avg_target):
    e_prediction_avg = e_prediction.mean()
    e_avg_loss = torch.sum((e_prediction_avg - e_avg_target)**2)
    return e_avg_loss

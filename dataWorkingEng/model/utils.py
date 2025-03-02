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

def pitch_loss(p_prediction, p_avg_target, p_std_target, x_mask):
    p_avg_predict_value = torch.sum(p_prediction, dim=1, keepdim=True) / torch.sum(x_mask.squeeze(1), dim=1, keepdim=True)
    p_avg_loss = torch.sum((p_avg_predict_value - p_avg_target)**2) / 15
    p_std_predic_value= torch.sqrt(torch.sum(((p_prediction - p_avg_predict_value) * x_mask.squeeze(1))**2, dim=1, keepdim=True) / torch.sum(x_mask.squeeze(1), dim=1, keepdim=True))
    # print(p_std_predic_value.shape, p_avg_predict_value.shape,x_mask.squeeze(1).shape, torch.sqrt(torch.sum(((p_prediction - p_avg_predict_value) * x_mask.squeeze(1))**2, dim=1, keepdim=True)).shape)
    p_std_loss = torch.sum((p_std_predic_value - p_std_target)**2) / 15
    # print(p_avg_loss, p_std_loss)
    return (p_avg_loss, p_std_loss)

def energy_loss(e_prediction, e_avg_target, x_mask):    # (B,L), (B,1), (B,1,L)
    # print(f'Shape e_prediction {e_prediction.shape}, Shape e_avg_target {e_avg_target.shape}, Shape x_mask {x_mask.shape}')
    # print(torch.sum(e_prediction, dim=1, keepdim=True) , torch.sum(x_mask.squeeze(1), dim=1, keepdim=True))
    # print(e_prediction[0], x_mask[0])
    e_predict_value = torch.sum(e_prediction, dim=1, keepdim=True) / torch.sum(x_mask.squeeze(1), dim=1, keepdim=True)
    e_avg_loss = torch.sum((e_predict_value - e_avg_target)**2) / 15
    return e_avg_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from my_utils import aud_len_from_mel


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()

        self.pad_size = (dilation * (kernel_size - 1))
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0)


    def forward(self, x):
        x = F.pad(x, (self.pad_size, 0), 'constant', 0)
        return self.conv(x)



class WaveNetLayer(nn.Module):
    def __init__(self, input_ch, skip_ch, layer_num):
        super(WaveNetLayer, self).__init__()

        self.dil_now = 2**(layer_num % 10)   # 1, 2, 4 -> 512

        self.W_f = CausalConv1d(input_ch, input_ch, kernel_size=2, dilation=self.dil_now)
        self.W_g = CausalConv1d(input_ch, input_ch, kernel_size=2, dilation=self.dil_now)
        self.V_f = nn.Conv1d(80, input_ch, kernel_size=1)
        self.V_g = nn.Conv1d(80, input_ch, kernel_size=1)

        self.skip_conv = nn.Conv1d(input_ch, skip_ch, kernel_size=1)
        self.resid_conv = nn.Conv1d(input_ch, input_ch, kernel_size=1)


    def forward(self, melspec, wav):

        z = torch.tanh(self.W_f(wav) + self.V_f(melspec)) \
            * \
            torch.sigmoid(self.W_g(wav) + self.V_g(melspec))

        skip_res = self.skip_conv(z)

        resid_res = self.resid_conv(z)
        resid_res = resid_res + wav

        return skip_res, resid_res



class WaveNet(nn.Module):
    def __init__(self, hidden_ch, skip_ch, num_layers, mu):
        super(WaveNet, self).__init__()

        self.skip_ch = skip_ch
        self.mu = mu
        #self.convtr = nn.ConvTranspose1d(in_channels=80, out_channels=80,
        #           kernel_size=512,   # 2 * 256 = 2 * hop_len
        #           stride=256,        # hop_len
        #           padding=256)       # ks // 2)
        self.embedding = CausalConv1d(1, hidden_ch, kernel_size=512)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(WaveNetLayer(hidden_ch, skip_ch, layer_num=i))
        self.out_conv = nn.Conv1d(skip_ch, mu, kernel_size=1)
        self.end_conv = nn.Conv1d(mu, mu, kernel_size=1)


    def forward(self, melspec, wav):

        melspec = torch.nn.functional.interpolate(melspec, aud_len_from_mel(melspec))[:, :, 1:]   #self.convtr(melspec)[:, :, 1:]
        wav = self.embedding(wav)

        skip_conn_res = torch.zeros((wav.size(0), self.skip_ch, wav.size(-1))).to(wav.device)
        for i in range(len(self.layers)):
            skip_one, wav = self.layers[i](melspec, wav)
            skip_conn_res = skip_conn_res + skip_one
        result_wav = self.end_conv(F.relu(
                                          self.out_conv(F.relu(skip_conn_res))
                                         ))
        return result_wav


    def inference(self, melspec):                 # bs=1
        new_wav_len = aud_len_from_mel(melspec)
        melspec = torch.nn.functional.interpolate(melspec, new_wav_len)[:, :, 1:]     #self.convtr(melspec)[:, :, 1:]

        whole_melspec = melspec
        melspec = melspec[:, :, :1]
        wav = torch.zeros((1, 1, 1)).to(melspec.device)
        for j in tqdm(range(2, new_wav_len+1)):
            new_wav = self.embedding(wav)

            skip_conn_res = torch.zeros((new_wav.size(0), self.skip_ch, new_wav.size(-1))).to(new_wav.device)
            for i in range(len(self.layers)):
                skip_one, new_wav = self.layers[i](melspec, new_wav)
                skip_conn_res = skip_conn_res + skip_one

            result_wav = self.end_conv(F.relu(
                                              self.out_conv(F.relu(skip_conn_res))
                                             ))
            result_wav = torch.argmax(result_wav, dim=1)
            # rewnew
            wav = torch.cat((wav, result_wav.unsqueeze(1)[:, : , -1:]), dim=-1)
            melspec = whole_melspec[:, :, :j]
        # return all but not the first fixed element 0
        return wav[:, :, 1:]

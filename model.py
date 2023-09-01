import torch
import torch.nn as nn
from typing import Optional, List
from conv_stft import ConvSTFT, ConviSTFT
from ft_lstm import FTLSTM, FTLSTM_RESNET
from complexnn import (
    complex_cat,
    complex_mask_multi,
    ComplexConv1d,
    ComplexConv2d,
    ComplexAttention,
    InstanceNorm,
    ComplexGateConvTranspose2d,
)


class MSDP_CRN(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        if nfft is None:
            self.nfft = int(2 ** torch.ceil(torch.log2(torch.tensor(self.nframe))))
        else:
            self.nfft = nfft
        self.fft_dim = self.nfft // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = ConvSTFT(nframe, nhop, self.nfft, out_feature="complex")
        self.istft = ConviSTFT(nframe, nhop, self.nfft, inp_feature="complex")

        self.conv_l = nn.ModuleList()

        self.encoder_l = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        self.atten_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.conv_l.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(5, 3),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(stride[idx], 1),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_l.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(5, 1),
                        padding=(2, 0),
                        stride=(stride[idx], 1),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.atten_l.append(
                ComplexAttention(
                    in_channel=self.cnn_num[idx + 1],
                    out_channel=self.cnn_num[idx + 1],
                    nfeat=nbin,
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(5, 1),
                            padding=(2, 0),
                            stride=(stride[-1 - idx], 1),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(5, 1),
                            padding=(2, 0),
                            stride=(stride[-1 - idx], 1),
                        ),
                        InstanceNorm(2 * self.fft_dim),
                        nn.PReLU(),
                    )
                )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )

        self.post_conv = ComplexConv1d(
            in_channels=2 * self.fft_dim,
            out_channels=2 * self.fft_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft(mic)  # [B, nfft+2, T], [B, (F_r, F_i), T]
        specs_ref = self.stft(ref)

        specs_mic_real, specs_mic_imag = (
            specs_mic[:, : self.fft_dim, :],
            specs_mic[:, self.fft_dim :, :],
        )

        mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_ref_real, specs_ref_imag = (
            specs_ref[:, : self.fft_dim, :],
            specs_ref[:, self.fft_dim :, :],
        )

        specs_mix = torch.stack(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        feat = torch.stack([specs_mic_real, specs_mic_imag], dim=1)

        x = specs_mix
        mask_store = []
        for idx, layer in enumerate(self.conv_l):
            # print("en", idx, x.shape)
            x = layer(x)  # x shape [B, C, F, T]
            # print("#", idx, x.shape)

            m = self.atten_l[idx](x)
            mask_store.append(m)
            # print("mask", idx, m.shape)
        # print("en", idx, x.shape)

        # print(specs_mix.shape)
        feat_store = []
        for idx, layer in enumerate(self.encoder_l):
            # print("feat", idx, feat.shape)
            feat = layer(feat)
            feat = feat * mask_store[idx]
            feat_store.append(feat)

        # print("feat", idx, feat.shape)
        nB, nC, nF, nT = x.shape
        x_r, x_i = torch.chunk(x, 2, dim=1)

        mask_r = self.rnns_r(x_r)
        mask_i = self.rnns_i(x_i)  # B,C,F,T

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        cmask = torch.concatenate([mask_r, mask_i], dim=1)

        feat_r, feat_i = complex_mask_multi(feat, cmask)

        feat = torch.concat([feat_r, feat_i], dim=1)

        # B,C,F,T
        for idx, layer in enumerate(self.decoder_l):
            feat = complex_cat([feat, feat_store[-idx - 1]], dim=1)
            # print("Tconv", idx, feat.shape)
            feat = layer(feat)
            # feat = feat[..., 1:]  # padding=(2, 0) will padding 1 col in Time dimension

        # print("Tconv", feat.shape)
        # B, 2, F, T -> B, F(r, i), T
        feat = feat.reshape(nB, self.fft_dim * 2, -1)  # B, F, T
        feat = self.post_conv(feat)

        out_wav = self.istft(feat)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


if __name__ == "__main__":
    from torchinfo import summary

    inp = torch.randn(2, 16000)

    model = MSDP_CRN(
        nframe=512,
        nhop=128,
        nfft=512,
        cnn_num=[16, 32, 64, 64, 128],
        stride=[2, 2, 1, 1, 1],
        rnn_hidden_num=128,
    )

    inp1 = torch.randn(1, 16000).cuda()
    inp2 = torch.randn(1, 16000).cuda()
    # flops, params = profile(model, (inp1, inp2))
    # flops /= 1e9
    # params /= 1e6
    # print(flops, params)
    model = model.cuda()
    est = model(inp1, inp2)
    print(est[0].shape, "loss_fn" in dir(model))
    summary(model, input_size=((1, 32000), (1, 32000)))
    # out = model(inp, inp)
    # for name, p in model.state_dict().items():
    #     if "conv_l" in name:
    #         print(name)

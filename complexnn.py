from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class InstanceNorm(nn.Module):
    def __init__(self, feats=1):
        """
        feats: CxF with input B,C,F,T
        """
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.register_parameter(
            "gamma", nn.Parameter(torch.ones(feats), requires_grad=True)
        )
        self.register_parameter(
            "beta", nn.Parameter(torch.zeros(feats), requires_grad=True)
        )

    def forward(self, inputs):
        """
        inputs shape is (B, C, F, T)
        """
        inputs = inputs.permute(0, 3, 1, 2)  # B, T, C, F
        nB, nT, nC, nF = inputs.shape
        # print(inputs.shape)
        inputs = inputs.reshape(nB, nT, -1)  # B, T, CxF

        mean = torch.mean(inputs, dim=-1, keepdim=True)
        var = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)

        std = torch.sqrt(var + self.eps)

        outputs = (inputs - mean) / std
        # print("                ", outputs.shape, self.gamma.shape)
        outputs = outputs * self.gamma + self.beta

        outputs = outputs.reshape(nB, nT, nC, nF)
        outputs = outputs.permute(0, 2, 3, 1)

        return outputs


def complex_mask_multi(inputs, mask, method="C"):
    """
    inputs, B, C(r, i), F, T
    mask, B, C(r, i), F, T
    """
    mask_r, mask_i = torch.chunk(mask, 2, dim=1)
    feat_r, feat_i = torch.chunk(inputs, 2, dim=1)
    if method == "E":
        mask_mag = F.sigmoid((mask_r**2 + mask_i**2) ** 0.5)
        mask_phs = torch.atan2(mask_i, mask_r)

        feat_mag = (feat_r**2 + feat_i**2) ** 0.5

        feat_phs = torch.atan2(feat_i, feat_r)
        feat_mag = feat_mag * mask_mag
        feat_phs += mask_phs
        feat_r = feat_mag * torch.cos(feat_phs)
        feat_i = feat_mag * torch.sin(feat_phs)
    elif method == "C":
        feat_r = feat_r * mask_r - feat_i * mask_i
        feat_i = feat_r * mask_i + feat_i * mask_r
    return feat_r, feat_i


def complex_cat(inputs, dim: int):
    """
    inputs: a list [inp1, inp2, ...]
    dim: the axis for complex features where real part first, imag part followed
    """
    real, imag = [], []

    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, dim)
        real.append(r)
        imag.append(i)

    real = torch.cat(real, dim)
    imag = torch.cat(imag, dim)

    output = torch.cat([real, imag], dim)
    return output


class ATTLayer(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.n_ch = n_ch // 2

        self.f_qk = nn.Sequential(
            nn.Conv2d(n_ch, self.n_ch * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.n_ch * 2),
            nn.PReLU(),
        )
        self.f_v = nn.Sequential(
            nn.Conv2d(n_ch, self.n_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.n_ch),
            nn.PReLU(),
        )

        self.t_qk = nn.Sequential(
            nn.Conv2d(n_ch, self.n_ch * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.n_ch * 2),
            nn.PReLU(),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(self.n_ch, n_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(n_ch),
            nn.PReLU(),
        )

    def forward(self, inp, x):
        """
        inp: B,C,F,T is the mic, lpb extracted feature
        x: B,C,F,T, is the mic extracted feature
        """
        f_qk = self.f_qk(inp)  # B,C,F,T
        vf = self.f_v(x)  # B,C,F,T
        # the kf F dimension must consistent with that of vf
        qf, kf = tuple(einops.rearrange(f_qk, "b (c k) f t->k b c f t", k=2))

        # b,t,f,c x b,t,c,y -> b,t,f,y
        f_score = torch.einsum("bcft,bcyt->btfy", qf, kf).contiguous() / (
            self.n_ch**0.5
        )
        f_score = f_score.softmax(dim=-1)
        # b,t,f,y x b,c,f2,t -> b,c,f2,t
        # fy x yc and add over y dim
        fout = torch.einsum("btfy,bcyt->bcft", f_score, vf).contiguous()

        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
        t_score = torch.einsum("bcft,bcfy->bfty", qt, kt).contiguous() / (
            self.n_ch**0.5
        )

        mask_v = -torch.finfo(t_score.dtype).max
        i, j = t_score.shape[-2:]
        mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
        t_score.masked_fill_(mask, mask_v)
        t_score = t_score.softmax(dim=-1)
        tout = torch.einsum("bfty,bcfy->bcft", t_score, fout).contiguous()
        out = self.proj(tout)
        return out + x


class TAttention(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.n_ch = n_ch // 2
        self.att_r = ATTLayer(self.n_ch)
        self.att_i = ATTLayer(self.n_ch)

    def forward(self, inp, x):
        inp_r, inp_i = torch.chunk(inp, 2, dim=1)
        x_r, x_i = torch.chunk(x, 2, dim=1)
        out_r = self.att_r(inp_r, x_r)
        out_i = self.att_i(inp_i, x_i)
        out = torch.concat([out_r, out_i], dim=1)
        return out


class ComplexAttention(nn.Module):
    def __init__(self, in_channel, out_channel, nfeat):
        super().__init__()

        self.atten_r = nn.Sequential(
            nn.Conv2d(
                # in_channels=out_channel * feat_num // 2,
                in_channels=in_channel // 2,
                out_channels=out_channel // 2,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            # nn.BatchNorm1d(out_channel // 2),
            InstanceNorm(out_channel // 2 * nfeat),
            nn.Sigmoid(),
        )
        self.atten_i = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel // 2,
                out_channels=out_channel // 2,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            # nn.BatchNorm1d(out_channel // 2),
            InstanceNorm(out_channel // 2 * nfeat),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """
        input dim should be [B, C, F, T]
        """
        real, imag = torch.chunk(input, 2, dim=1)

        # real = self.conv_r(real)
        # imag = self.conv_i(imag)
        # B, C, F, T = real.size()

        # real = real.reshape(B, C * F, T)
        # imag = imag.reshape(B, C * F, T)

        real = self.atten_r(real)
        imag = self.atten_i(imag)

        return torch.cat([real, imag], dim=1)


class ComplexGate(nn.Module):
    def __init__(self, in_channels: int, dim: int = 1):
        super().__init__()
        self.complex_dim = dim
        in_channels = in_channels // 2
        out_c = in_channels // 2

        self.conv_rr = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_c,
            kernel_size=(3, 1),
            padding=(1, 0),
            stride=(1, 1),
        )

        self.conv_ii = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_c,
            kernel_size=(3, 1),
            padding=(1, 0),
            stride=(1, 1),
        )

    def forward(self, inputs):
        r, i = torch.chunk(inputs, 2, dim=self.complex_dim)

        rr_out = self.conv_rr(r)
        ii_out = self.conv_ii(i)

        real = F.sigmoid(rr_out)
        imag = F.sigmoid(ii_out)

        out = torch.cat([real, imag], dim=self.complex_dim)

        return out


class ComplexPReLU(nn.Module):
    def __init__(self, complex_dim: int):
        super().__init__()
        self.prelu_r = nn.PReLU()
        self.prelu_i = nn.PReLU()
        self.complex_dim = complex_dim

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_dim)
        real = self.prelu_r(real)
        imag = self.prelu_i(imag)

        out = torch.cat([real, imag], dim=self.complex_dim)
        return out


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        causal: bool = True,
        complex_dim=1,
    ):
        """
        in_channels: contains real and image, which use different conv2d to process repectively
        causal: only padding the time dimension, left side, if causal=True, otherwise padding both
        padding: padding[0] for Frequency dimension, [1] for Time dimension as input shape is [B, 2, F, T],
                    padding Time for the requirement that need to do convolution along the frame dimension.

        """
        super().__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.causal = causal
        self.padding = padding
        self.complex_dim = complex_dim

        self.conv_r = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
        )

        self.conv_i = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, inputs):
        """
        inputs shape should be B, F, T
        output [B, F, T]
        """

        # * padding zeros at Time dimension as Convolution kernel may be (x, 2) where 2 for Time dimension.
        if self.causal and self.padding:
            inputs = F.pad(inputs, (self.padding, 0))
        else:
            inputs = F.pad(inputs, (self.padding, self.padding))

        # NOTE need to split the input manually, because the channel dimension of input data is combined with Frequence dimension.
        # complex_dim == 0 means the input shape is (B, T, 2xF) or (T, 2xF) where 2xF indicate (Fr,..,Fr,Fi,...Fi)
        if self.complex_dim == 0:
            real = self.conv_r(inputs)
            imag = self.conv_i(inputs)
            rr, ir = torch.chunk(real, 2, self.complex_dim)
            ri, ii = torch.chunk(imag, 2, self.complex_dim)

        else:
            # * split inputs to 2 groups along complex_dim axis
            real, imag = torch.chunk(inputs, 2, self.complex_dim)  # B, C, F, T
            rr = self.conv_r(real)
            ii = self.conv_i(imag)
            ri = self.conv_i(real)
            ir = self.conv_r(imag)

        real = rr - ii
        imag = ri + ir
        out = torch.cat([real, imag], self.complex_dim)

        return out


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal: bool = True,
        complex_dim=1,
    ):
        """
        in_channels: contains real and image, which use different conv2d to process repectively
        causal: only padding the time dimension, left side, if causal=True, otherwise padding both
        padding: padding[0] for Frequency dimension, [1] for Time dimension as input shape is [B, 2, F, T],
                    padding Time for the requirement that need to do convolution along the frame dimension.

        """
        super().__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.causal = causal
        self.padding = padding
        self.complex_dim = complex_dim

        self.conv_r = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=(padding[0], 0),
            dilation=dilation,
            groups=groups,
        )

        self.conv_i = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=(padding[0], 0),
            dilation=dilation,
            groups=groups,
        )

        nn.init.normal_(self.conv_r.weight.data, mean=0.0, std=0.05)
        nn.init.constant_(self.conv_r.bias.data, 0.0)
        nn.init.normal_(self.conv_i.weight.data, mean=0.0, std=0.05)
        nn.init.constant_(self.conv_i.bias.data, 0.0)

    def forward(self, inputs):
        """
        inputs shape should be [B, 2, F, T], or [2, F, T], where 2 for complex (real, imag)
        output [B, 2, F_out, T_out]
        """

        # * padding zeros at Time dimension as Convolution kernel may be (x, 2) where 2 for Time dimension.
        if self.causal and self.padding[1] != 0:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        # NOTE need to split the input manually, because the channel dimension of input data is combined with Frequence dimension.
        # complex_dim == 0 means the input shape is (B, T, 2xF) or (T, 2xF) where 2xF indicate (Fr,..,Fr,Fi,...Fi)
        if self.complex_dim == 0:
            real = self.conv_r(inputs)
            imag = self.conv_i(inputs)
            rr, ir = torch.chunk(real, 2, self.complex_dim)
            ri, ii = torch.chunk(imag, 2, self.complex_dim)

        else:
            # * split inputs to 2 groups along complex_dim axis
            real, imag = torch.chunk(inputs, 2, self.complex_dim)
            rr = self.conv_r(real)
            ii = self.conv_i(imag)
            ri = self.conv_i(real)
            ir = self.conv_r(imag)

        # if torch.isnan(rr).int().sum():
        #     print(
        #         "complex cnn rr is nan",
        #         torch.isnan(self.conv_r.state_dict()["weight"]).sum(),
        #         torch.isnan(self.conv_r.state_dict()["bias"]).sum(),
        #     )
        real = rr - ii
        imag = ri + ir
        out = torch.cat([real, imag], self.complex_dim)

        return out


class NavieComplexLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=False,
        projection_dim=None,
        bidirectional=False,
    ):
        super().__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.batch_first = batch_first

        self.lstm_r = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.rnn_units,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        self.lstm_i = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.rnn_units,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1

        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.trans_r = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
            self.trans_i = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
        else:
            self.projection_dim = None

    def flatten_parameters(self):
        self.lstm_r.flatten_parameters()
        self.lstm_i.flatten_parameters()

    def forward(self, inputs) -> List:
        """
        inputs: a list (real_l, imag_l) where each element shape is [T, B, -1] or [B, T, -1]
        output: a list (real, imag)
        """
        real, imag = inputs

        rr, (h_rr, c_rr) = self.lstm_r(real)
        ir, (h_ir, c_ir) = self.lstm_r(imag)
        ri, (h_ri, c_ri) = self.lstm_i(real)
        ii, (h_ii, c_ii) = self.lstm_i(imag)

        real, imag = rr - ii, ri + ir

        if self.projection_dim is not None:
            real = self.trans_r(real)
            imag = self.trans_i(imag)

        return [real, imag]


class ComplexGateConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        groups=1,
        causal=False,
        complex_dim=1,
    ):
        """
        in_channels: real + imag
        out_channels: real + imag
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_dim = complex_dim
        self.causal = causal

        self.tc1 = nn.Sequential(
            ComplexConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                complex_dim=complex_dim,
                groups=groups,
            ),
            nn.Sigmoid(),
        )

        self.tc2 = ComplexConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            complex_dim=complex_dim,
            groups=groups,
        )

    def forward(self, inputs):
        mask = self.tc1(inputs)
        out = self.tc2(inputs)
        out = mask * out
        return out


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        groups=1,
        causal=False,
        complex_dim=1,
    ):
        """
        in_channels: real + imag
        out_channels: real + imag
        """
        super().__init__()

        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.complex_dim = complex_dim
        self.causal = causal

        self.convT_r = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.convT_i = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

        # nn.init.normal_(self.convT_r.weight.data, std=0.05)
        # nn.init.normal_(self.convT_i.weight.data, std=0.05)
        # nn.init.constant_(self.convT_r.bias.data, 0)
        # nn.init.constant_(self.convT_i.bias.data, 0)

    def forward(self, inputs):
        # if isinstance(inputs, torch.Tensor):
        #     real, imag = torch.chunk(inputs, 2, self.complex_dim)
        # elif isinstance(inputs, (tuple, list)):
        #     real, imag = inputs[0], inputs[1]

        if self.complex_dim == 0:
            real = self.convT_r(inputs)
            imag = self.convT_i(inputs)

            rr, ir = torch.chunk(real, 2, self.complex_dim)
            ri, ii = torch.chunk(imag, 2, self.complex_dim)
        else:
            real, imag = torch.chunk(inputs, 2, self.complex_dim)
            rr, ir = self.convT_r(real), self.convT_r(imag)
            ri, ii = self.convT_i(real), self.convT_i(imag)

        real = rr - ii
        imag = ir + ri

        out = torch.cat([real, imag], self.complex_dim)

        return out


# TODO not complete
# class ComplexBN(nn.Module):
#     def __init__(
#         self,
#         num_features,
#         eps=1e-5,
#         momentum=0.1,
#         affine=True,
#         track_running_states=True,
#         complex_dim=1,
#     ):
#         super().__init__()

#         self.num_features = num_features
#         self.eps = eps
#         self.complex_dim = complex_dim
#         self.tracking_running_states = track_running_states
#         self.affine = affine

#         if affine:
#             self.Wrr = nn.Parameter(torch.Tensor(self.num_features))
#             self.Wri = nn.Parameter(torch.Tensor(self.num_features))
#             self.Wii = nn.Parameter(torch.Tensor(self.num_features))
#             self.Br = nn.Parameter(torch.Tensor(self.num_features))
#             self.Bi = nn.Parameter(torch.Tensor(self.num_features))
#         else:
#             self.register_parameter("Wrr", None)
#             self.register_parameter("Wri", None)
#             self.register_parameter("Wii", None)
#             self.register_parameter("Br", None)
#             self.register_parameter("Bi", None)

#         if track_running_states:
#             self.register_buffer("RMr", torch.zeros(self.num_features))
#             self.register_buffer("RMi", torch.zeros(self.num_features))
#             self.register_buffer("RVrr", torch.ones(self.num_features))
#             self.register_buffer("RVri", torch.zeros(self.num_features))
#             self.register_buffer("RVii", torch.ones(self.num_features))
#             self.register_buffer(
#                 "num_bactches_tracked", torch.tensor(0, dtype=torch.long)
#             )
#         else:
#             self.register_buffer("RMr", None)
#             self.register_buffer("RMi", None)
#             self.register_buffer("RVrr", None)
#             self.register_buffer("RVri", None)
#             self.register_buffer("RVii", None)
#             self.register_buffer("num_bactches_tracked", None)

#         self.reset_parameters()

#     def reset_running_states(self):
#         if self.tracking_running_states:
#             self.RMr.zero_()
#             self.RMi.zero_()
#             self.RVrr.fill_(1)
#             self.RVri.zero_()
#             self.RVii.fill_(1)
#             self.num_batches_tracked.zero_()

#     def reset_parameters(self):
#         self.reset_running_states()
#         if self.affine:
#             pass


if __name__ == "__main__":
    inpr = torch.randn(5, 2, 3)
    inpi = torch.randn(5, 2, 3)
    model = NavieComplexLSTM(6, 20)
    out = model([inpr, inpi])  # ([5,2,10], [5,2,10])
    print(out[0].shape)

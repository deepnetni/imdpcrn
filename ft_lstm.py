import torch
import torch.nn as nn


class FTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        """
        Args:
            input_size: should be equal to C of input shape B,C,F,T
        """
        super().__init__()

        self.f_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            batch_first=batch_first,
            bidirectional=True,
        )

        self.f_post = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
        )

        self.t_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        self.t_post = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input shape should be B,C,F,T
        """
        nB, nC, nF, nT = x.shape
        x = x.permute(0, 3, 2, 1)  # B, T, F, C
        x = x.reshape(-1, nF, nC)  # BxT,F,C
        x, _ = self.f_unit(x)  # BxT,F,C
        x = self.f_post(x)

        x = x.reshape(nB, nT, nF, nC)
        x = x.permute(0, 2, 1, 3)  # B,F,T,C
        x = x.reshape(-1, nT, nC)  # BxF,T,C

        x, _ = self.t_unit(x)
        x = self.t_post(x)
        x = x.reshape(nB, nF, nT, nC)
        x = x.permute(0, 3, 1, 2)  # B,C,F,T

        return x


class FTLSTM_RESNET(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        """
        Args:
            input_size: should be equal to C of input shape B,C,F,T
        """
        super().__init__()

        self.f_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            batch_first=batch_first,
            bidirectional=True,
        )

        self.f_post = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
        )

        self.t_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        self.t_post = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
        )

    def forward(self, inp: torch.Tensor):
        """
        Args:
            x: input shape should be B,C,F,T
        """
        nB, nC, nF, nT = inp.shape
        x = inp.permute(0, 3, 2, 1)  # B, T, F, C
        x = x.reshape(-1, nF, nC)  # BxT,F,C
        x, _ = self.f_unit(x)  # BxT,F,C
        x = self.f_post(x)

        x = x.reshape(nB, nT, nF, nC)
        x = x.permute(0, 3, 2, 1)  # B,C,F,T
        inp = inp + x

        x = inp.permute(0, 2, 3, 1)  # B,F,T,C
        x = x.reshape(-1, nT, nC)  # BxF,T,C
        x, _ = self.t_unit(x)
        x = self.t_post(x)
        x = x.reshape(nB, nF, nT, nC)
        x = x.permute(0, 3, 1, 2)  # B,C,F,T
        x += inp

        return x

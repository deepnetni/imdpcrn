import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.signal import get_window


def init_conv_stft_kernels(nframe: int, nfft: int, win_type=None, inverse=False):
    if win_type is None:
        win = np.ones(nframe)
    else:
        if win_type == "hann sqrt":
            win = get_window(
                "hann", nframe, fftbins=True
            )  # fftbins=True, win is not symmetric

            win = np.sqrt(win)
        else:
            win = get_window(win_type, nframe, fftbins=True)

    N = nfft

    # * the fourier_baisis is nframe, N//2 + 1
    # [[ W_N^0x0, W_N^0x1, ..., W_N^0x(N-1) ]
    #  [ W_N^1x0, W_N^1x1, ..., W_N^1x(N-1) ]
    #  [ W_N^2x0, W_N^2x1, ..., W_N^2x(N-1) ]]
    fourier_basis = np.fft.rfft(np.eye(N))[:nframe]
    # print(fourier_basis.shape)  # 400, 257
    # * (nframe, nfft // 2 + 1)
    kernel_r, kernel_i = np.real(fourier_basis), np.imag(fourier_basis)

    # * reshape to (2 x (nfft // 2 + 1), nframe)
    kernel = np.concatenate([kernel_r, kernel_i], axis=1).T
    # print(kernel.shape)         # (514, 400)

    if inverse:
        # * A dot pinv(A) = I
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * win
    # * kernel is (out_channel, inp_channel, kernel_size)
    kernel = kernel[:, None, :]  # (2 x (nfft // 2 + 1), 1, nframe)

    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        win[None, :, None].astype(np.float32)
    )


class ConvSTFT(nn.Module):
    def __init__(
        self,
        nframe: int = 512,
        nhop: int = 128,
        nfft: int = None,
        win_type: str = "hann",
        out_feature: str = "magPhase",
    ):
        """
        Args:
            out_feature: "magPhase"(default), or "complex"

        Return:
            1. B, 2xF(real, imag), T if `out_feature` == "complex";
            2. Magnitude, Pahse with shape B, F, T by default;
            where F = nfft // 2 + 1

        Example:
            >>> x = torch.randn(10, 16000) # B, T
            >>> stft = ConvSTFT(512, 256, 512, "hann", "complex")
            >>> Xk = stft(x)
            >>>
            >>> stft = ConvSTFT(512, 256, 512, "hann", "magPhase")
            >>> Xmag, Xphase = stft(x)
        """
        super().__init__()

        self.nframe = nframe
        self.nhop = nhop
        self.out_feature = out_feature

        if nfft is None:
            # * rounding up to an exponential multiple of 2
            self.nfft = int(2 ** np.ceil(np.log2(nframe)))
        else:
            self.nfft = nfft

        kernel, _ = init_conv_stft_kernels(nframe, self.nfft, win_type, False)

        self.register_buffer("weight", kernel)

    def forward(self, x):
        """
        x shape should be: [ B, 1, T ] or [ B, T ]
        output shape is [B, 2xF(Fr, Fi), T] if complex or (mags, phase) with shape [B, F, T] default

        Note: 2xF dimension is composed by F_r, ..., F_r, F_i, ..., F_i
        """
        if x.dim() == 2:
            # * expand shape to (:, 1, :)
            x = torch.unsqueeze(x, dim=1)

        x = F.pad(x, (self.nframe - self.nhop, self.nframe - self.nhop))
        # * self.weight shape is [ 2 x (nfft//2 + 1), 1, nframe ]
        out_complex = F.conv1d(x, self.weight, stride=self.nhop)

        if self.out_feature == "complex":
            return out_complex  # B, F, T

        dim = self.nfft // 2 + 1
        real = out_complex[:, :dim, :]
        imag = out_complex[:, dim:, :]

        mags = torch.sqrt(real**2 + imag**2)
        phas = torch.atan2(imag, real)

        return mags, phas


class ConviSTFT(nn.Module):
    def __init__(
        self,
        nframe,
        nhop,
        nfft: int = None,
        win_type: str = "hann",
        inp_feature="magPhase",
    ):
        """torch based istft implementation
        Args:
           inp_feature: "magPhase" or "complex"

        Return:
           B, T

        Example:
            >>> stft = ConvSTFT(512, 256, 512, "hann", "complex")
            >>> istft = ConviSTFT(512, 256, 512, "hann", "complex")
            >>> Xk = stft(x)
            >>> data = istft(Xk)
            >>>
            >>> stft = ConvSTFT(512, 256, 512, "hann", "magPhase")
            >>> istft = ConviSTFT(512, 256, 512, "hann", "magPhase")
            >>> Xmag, Xphase = stft(x)
            >>> data = istft(Xmag, Xphase)
        """
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.inp_feature = inp_feature

        if nfft is None:
            # * rounding up to an exponential multiple of 2
            self.nfft = int(2 ** np.ceil(np.log2(nframe)))
        else:
            self.nfft = nfft

        kernel, win = init_conv_stft_kernels(nframe, self.nfft, win_type, True)
        self.register_buffer("weight", kernel)
        self.register_buffer("win", win)
        self.register_buffer("enframe", torch.eye(nframe)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs: [B, N+2, T] (complex spec) if self.inp_features = 'complex'
        inputs, phase: [B, N//2+1, T], [B, N//2 + 1] if self.inp_features = 'MagPhase'
        outputs: [B, 1, T]
        """

        if self.inp_feature == "magPhase" and phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)

            inputs = torch.cat([real, imag], dim=1)

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.nhop)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.win.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.nhop)
        outputs = outputs / (coff + 1e-8)

        outputs_ = outputs[..., (self.nframe - self.nhop) : -(self.nframe - self.nhop)]

        return outputs_


def verify_fft():
    import librosa

    nfft = 512
    nframe = 400
    nhop = 100

    # * B, 1, T, where 1 for channel
    inputs = torch.randn([1, 1, 16000])
    np_inputs = inputs.numpy().reshape(-1)

    fft = ConvSTFT(nframe, nhop, nfft, "hann")

    out1 = fft(inputs)[0]  # M, P get magnitude
    out1 = out1.numpy()[0]  # B, F, T
    # NOTE torch and librosa output don't contain the first and last padding frame
    out1 = out1[:, 1:-1]

    librosa_stft = librosa.stft(
        np_inputs,
        win_length=nframe,
        n_fft=nfft,
        hop_length=nhop,
        window="hann",
        center=True,
    )

    librosa_istft = librosa.istft(
        librosa_stft,
        hop_length=nhop,
        win_length=nframe,
        n_fft=nfft,
        window="hann",
        center=True,
    )

    win = torch.from_numpy(get_window("hann", nframe, fftbins=True)).float()
    torch_stft = torch.stft(
        inputs.flatten(),
        n_fft=nfft,
        hop_length=nhop,
        win_length=nframe,
        window=win,
        center=True,
        pad_mode="constant",
        onesided=True,
        return_complex=True,
    )
    torch_stft = torch_stft.numpy()

    print(out1.shape, librosa_stft.shape, librosa_istft.shape, torch_stft.shape)
    # * compare torch.stft and librosa.stft
    print(np.mean((np.abs(torch_stft) - np.abs(librosa_stft)) ** 2))
    # * compare torch.stft and ConvSTFT
    print(np.mean((out1 - np.abs(librosa_stft)) ** 2))


def verify_ifft_complex():
    data = np.random.randn(16000 * 8)[None, None, :]
    inp = torch.from_numpy(data.astype(np.float32))

    fft = ConvSTFT(400, 100, None, "hann", "complex")
    ifft = ConviSTFT(400, 100, None, "hann", "complex")

    out1 = fft(inp)
    wav = ifft(out1)

    print(inp.shape, wav.shape, out1.shape)

    diff = torch.mean(torch.abs(inp[..., : wav.size(2)] - wav) ** 2)
    print(diff)


def verify_ifft_mag():
    data = np.random.randn(16000 * 8)[None, None, :]
    inp = torch.from_numpy(data.astype(np.float32))

    fft = ConvSTFT(400, 100, None, "hann")
    ifft = ConviSTFT(400, 100, None, "hann")

    mag, pha = fft(inp)

    wav = ifft(mag, pha)

    diff = torch.mean(torch.abs(inp[..., : wav.size(2)] - wav) ** 2)
    print(diff)


if __name__ == "__main__":
    # verify_fft()
    verify_ifft_complex()
    verify_ifft_mag()

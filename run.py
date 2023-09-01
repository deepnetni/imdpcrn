import os
import shutil
import sys
import argparse
import torch
from pathlib import Path
from model import MSDP_CRN
import soundfile as sf
import numpy as np
from gcc_phat import gcc_phat


def fread(fname, sub_mean=True):
    data, fs = sf.read(fname)

    if not os.path.exists(fname):
        raise RuntimeError(f"file not exist: {fname}")

    if sub_mean:
        try:
            data = data - np.mean(data, axis=0, keepdims=True)
        except Exception as e:
            print("########", e)

    data = np.clip(data, -1.0, 1.0)
    return data.astype(np.float32)


@torch.no_grad()
def predict(lpb_p, mic_p, out_p: str = ""):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(out_p) is False:
        if out_p != "":
            os.makedirs(out_p)
    else:
        shutil.rmtree(out_p)
        os.makedirs(out_p)

    net = MSDP_CRN(
        nframe=512,
        nhop=128,
        nfft=512,
        cnn_num=[16, 32, 64, 128],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=128,
    )

    ckpt = torch.load("model.pth")
    net.load_state_dict(ckpt)
    net.to(device)
    net.eval()

    ref = fread(lpb_p)
    mic = fread(mic_p)

    tau, _ = gcc_phat(mic, ref, fs=16000, interp=1)
    tau = max(0, int((tau - 0.001) * 16000))
    ref = np.concatenate([np.zeros(tau), ref], axis=-1)[: mic.shape[-1]]

    ref = torch.from_numpy(ref.astype(np.float32))
    mic = torch.from_numpy(mic)

    ref = ref.to(device).reshape(1, -1)  # (1, T)
    mic = mic.to(device).reshape(1, -1)

    with torch.no_grad():
        est = net(mic, ref)

    est = est.cpu().float().numpy()
    est = est.squeeze()

    _, out_name = os.path.split(lpb_p)
    if out_p != "":
        fout = os.path.join(out_p, out_name)
    else:
        fout = out_name

    try:
        sf.write(fout, est, 16000)
    except Exception as e:
        dir_p, _ = os.path.split(fout)
        if not os.path.exists(dir_p):
            os.makedirs(dir_p)
            sf.write(fout, est, 16000)
        else:
            print("##", e)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpb", help="reference wav file path")
    parser.add_argument("--mic", help="microphone wav file path")
    parser.add_argument("--out", help="output file path", default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    lpb_p = args.lpb
    mic_p = args.mic
    out_p = args.out

    predict(lpb_p, mic_p, out_p)

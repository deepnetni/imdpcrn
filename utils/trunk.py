import logging
import os
import random
import re
import sys
from pathlib import Path
import librosa

sys.path.append(str(Path(__file__).parent.parent))
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from gcc_phat import gcc_phat

from utils.logger import getLogger


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
    return data.astype(np.float32), fs


class _Dataset:
    def __init__(self):
        self._eps = np.finfo(np.float32).eps

    def _check_power(self, data: np.ndarray, threshold: int = -40) -> bool:
        try:
            power = 10 * np.log10(np.mean(data**2) + self._eps)
        except Exception as e:
            print("#####", e)
            return False

        return bool(power > threshold)

    def fread(self, fname, sub_mean=True):
        data, fs = sf.read(fname)

        if not os.path.exists(fname):
            raise RuntimeError(f"file not exist: {fname}")

        if sub_mean:
            try:
                data = data - np.mean(data, axis=0, keepdims=True)
            except Exception as e:
                print("########", e)

        data = np.clip(data, -1.0, 1.0)
        return data.astype(np.float32), fs

    def fwrite(self, fname: str, data: np.ndarray, fs: int = 16000):
        sf.write(fname, data, fs)

    def fwrite_w_idx(self, fpath: str, index: str, data: np.ndarray, fs: int = 16000):
        fname, suffix = os.path.split(fpath)
        fname = f"{fname}_{index}{suffix}"
        sf.write(fname, data, fs)


class BlindTrunk(_Dataset):
    def __init__(self, dirname, out_dir: str = "", align: bool = False):
        super().__init__()
        self.dirname = Path(dirname)
        self.f_list = []
        self.align = align
        self._scan("**/*lpb.wav", ["lpb", "mic"])
        self.out_dir = out_dir

    def _scan(self, key: str = "**/*lpb.wav", map: List = ["lpb", "mic"]):
        self.map = map
        for ref_p in self.dirname.glob(key):
            ref_path = str(ref_p)
            mic_path = ref_path.replace(*map)
            if os.path.exists(mic_path):
                self.f_list.append((ref_path, mic_path))

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, index):
        ref_path, mic_path = self.f_list[index]

        ref, fs_ref = self.fread(ref_path)
        mic, fs_mic = self.fread(mic_path)

        if fs_ref != fs_mic:
            raise RuntimeError("the fs of lpb and mic are not equal.")

        if self.align is True:
            tau, _ = gcc_phat(mic, ref, fs=fs_ref, interp=1)
            tau = max(0, int((tau - 0.001) * fs_ref))
            ref = np.concatenate([np.zeros(tau), ref], axis=-1)[: mic.shape[-1]]

        ref = torch.from_numpy(ref)
        mic = torch.from_numpy(mic)

        return ref, mic

    def __iter__(self):
        self.pick_idx = 0
        return self

    def __next__(self):
        if self.pick_idx < len(self.f_list):
            ref_path, mic_path = self.f_list[self.pick_idx]
            ref, fs_ref = self.fread(ref_path)
            mic, fs_mic = self.fread(mic_path)

            # if fs_ref != 16000:
            #     ref = librosa.resample(ref, orig_sr=fs_ref, target_sr=16000)
            # if fs_mic != 16000:
            #     mic = librosa.resample(mic, orig_sr=fs_mic, target_sr=16000)

            if fs_ref != fs_mic:
                raise RuntimeError("the fs of lpb and mic are not equal.")

            N1, N2 = ref.shape[-1], mic.shape[-1]
            N = min(N1, N2)
            ref, mic = ref[..., :N], mic[..., :N]

            if self.align is True:
                tau, _ = gcc_phat(mic, ref, fs=fs_ref, interp=1)
                tau = max(0, int((tau - 0.001) * fs_ref))
                ref = np.concatenate([np.zeros(tau), ref], axis=-1, dtype=np.float32)[
                    :N
                ]

            ref = torch.from_numpy(ref)[None, :]
            mic = torch.from_numpy(mic)[None, :]

            mic_p, mic_name = os.path.split(mic_path)
            fout = mic_name.replace(self.map[1], "enh")
            fout = os.path.join(mic_p, fout)
            if self.out_dir != "":
                fout = fout.replace(str(self.dirname), self.out_dir)

            self.pick_idx += 1

            return ref, mic, fout
        else:
            raise StopIteration


class AECTrunk(_Dataset):
    def __init__(self, dirname, samples: Dict, seed: Optional[int] = None):
        super().__init__()

        self.dirname = Path(dirname)
        self.d_synt = self.dirname.joinpath("synthetic")
        self.d_real = self.dirname.joinpath("real")
        self.fs = 16000
        self.log = getLogger("AECTrunk", mode="console", level=logging.INFO)

        flist = self._analysis()
        n_flist = len(flist)

        if seed is not None:
            random.seed(seed)

        random.shuffle(flist)
        # for f in flist:
        #     print(f)

        self.out_dict = {}

        st = 0
        for k, v in samples.items():
            # print(isinstance(v, int), isinstance(v, float))
            if isinstance(v, int):
                num = v
            else:
                num = int(v * n_flist)

            self.out_dict[k] = flist[st : st + num]
            st += num
            self.log.info(
                f"{k} get {len(self.out_dict[k])} {len(self.out_dict[k])/360:.2f}h"
            )

    def _analysis(self):
        """
        search the echo wavfiles under given aec directory.
        """
        out = []
        synt = list(self.d_synt.glob("echo_signal/*.wav"))
        real = list(self.d_real.glob("*farend_singletalk*_mic.wav"))

        for f in real:
            fname = f.name.replace("mic", "lpb")
            ref_p = f.parent.joinpath(fname)
            if not ref_p.exists():
                real.remove(f)
            else:
                out.append((ref_p, f))

        for f in synt:
            ref_dir = f.parent.parent.joinpath("farend_speech")
            ref_name = f.name.replace("echo", "farend_speech")
            ref_p = ref_dir.joinpath(ref_name)
            out.append((ref_p, f))

        return out

    @property
    def ref_mic_dict(self):
        return self.out_dict


class NSTrunk(_Dataset):
    def __init__(self, dirname: str, samples: Dict, seed: Optional[int] = None):
        super().__init__()
        self.dir = Path(dirname)
        self.d_clean = self.dir.joinpath("clean")
        self.d_noise = self.dir.joinpath("noise")
        self.samples = samples
        wav_clean_l, wav_noise_l = self._analysis()
        self.log = getLogger("NSTrunk", mode="console", level=logging.INFO)

        if seed is not None:
            random.seed(seed)

        random.shuffle(wav_clean_l)
        random.shuffle(wav_noise_l)

        self.cln_dict = {}
        self.nis_dict = {}
        st_sph, st_nis = 0, 0
        for k, n in samples.items():
            if isinstance(n, int):
                n_sph = n
                n_nis = n * 3
            else:
                n_sph = int(n * len(wav_clean_l))
                n_nis = int(n * len(wav_noise_l) * 3)

            self.cln_dict[k] = wav_clean_l[st_sph : st_sph + n_sph]
            self.nis_dict[k] = wav_noise_l[st_nis : st_nis + n_nis]

            st_sph += n_sph
            st_nis += n_nis
            self.log.info(
                f"Extract {len(self.cln_dict[k]): <6} sph samples {len(self.cln_dict[k])/120:.2f}h for {k}"
            )
            self.log.info(
                f"Extract {len(self.nis_dict[k]): <6} nis samples {len(self.nis_dict[k])/360:.2f}h for {k}"
            )

    def _analysis(self):
        """
        search the echo wavfiles under given aec directory.
        """
        wav_clean_l = list(self.d_clean.glob("*.wav"))
        wav_noise_l = list(self.d_noise.glob("*.wav"))

        return wav_clean_l, wav_noise_l

    @property
    def speech_dict(self):
        return self.cln_dict

    @property
    def noise_dict(self):
        return self.nis_dict


##################
# torch dataset  #
##################


class NSTrunkDset(Dataset):
    """
    Used for datasets generated by DNS challenge script.
    return:
        noisy, clean wav data.
    """

    def __init__(self, noisy_dset_p: str, clean_dset_p: str):
        super().__init__()
        self.noisy_p = Path(noisy_dset_p)
        self.clean_p = Path(clean_dset_p)
        self.noisy_clean_l = self._scan()

    def _scan(self):
        noisy_clean_l = []
        # clean_wav_p_l = list(self.clean_p.glob("*.wav"))
        for noisy_wav_p in self.noisy_p.glob("*.wav"):
            pat = re.search("fileid_[0-9]*.wav", str(noisy_wav_p))
            if pat is None:
                continue

            clean_f = f"clean_{pat.group()}"
            clean_p = self.clean_p / clean_f
            # if clean_p in clean_wav_p_l:
            if clean_p.exists():
                noisy_clean_l.append((noisy_wav_p, clean_p))

        return noisy_clean_l

    def __len__(self):
        return len(self.noisy_clean_l)

    def __getitem__(self, index):
        noisy_p, clean_p = self.noisy_clean_l[index]
        noisy_d, fs = fread(noisy_p)
        clean_d, _ = fread(clean_p)

        return noisy_d, clean_d


if __name__ == "__main__":
    nis_d = "/home/ll/datasets/ns/noisy"
    cln_d = "/home/ll/datasets/ns/clean"

    dset = NSTrunkDset(nis_d, cln_d)

    # blind_p = "/home/deepnetni/datasets/blind_test_set"
    # bset = BlindTrunk(blind_p)

    # for ref, mic, fout in bset:
    #     pass

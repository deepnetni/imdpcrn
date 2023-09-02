import os
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))

from itertools import repeat

from utils.AECMOS.AECMOS_local.aecmos import AECMOSEstimator
from utils.parallel import Parallel
from pesq import pesq
from tqdm import tqdm
from utils.audiolib import audioread

fpath = Path(__file__).parent.parent
mos_p = fpath / "utils/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx"
mos_p_48k = fpath / "utils/AECMOS/AECMOS_local/Run_1668423760_Stage_0.onnx"


class _Score:
    def __init__(self, *args):
        self.state = {}
        for k in args:
            self.state[k] = {"val": 0, "num": 0}

    def update(self, k, v):
        record = self.state[k]
        record["num"] += 1
        record["val"] += v

    def score(self, k):
        record = self.state[k]

        val = record["val"]
        num = record["num"]

        return round(val / num, 3) if num != 0 else 0

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        keys = list(self.state.keys())
        if self.iter_idx < len(keys):
            k = keys[self.iter_idx]
            self.iter_idx += 1
            return k, self.score(k)
        else:
            raise StopIteration


def work_pesq(fname_list, fs):
    sph_fname, est_fname = fname_list
    sph, sr_sph = audioread(sph_fname)
    est, sr_est = audioread(est_fname)
    assert (
        sr_sph == sr_est
    ), f"sample rate of est and sph are not equal {sr_est}!={sr_sph}"
    assert sr_sph == fs, "sample rate of sph and configure are not equal"

    score = pesq(fs, sph, est, "wb" if fs == 16000 else "nb")
    return score


def calculate_pesq_score(
    src_path: str,
    est_path: str,
    est_suffix: str = "enh",
    sample_rate: int = 16000,
):
    # search result by src
    src_p = Path(src_path)
    flist = []
    print(f"## source path {src_path}, dest path {est_path}")

    for sph_fname in src_p.glob("**/*sph.wav"):
        est_fname = str(sph_fname).replace(src_path, est_path)
        est_fname = est_fname.replace("sph", est_suffix)
        if os.path.exists(est_fname):
            flist.append([sph_fname, est_fname])

    p = Parallel()
    scores = p.add(
        "pesq", work_pesq, args=list(zip(flist, repeat(sample_rate))), show=False
    )
    p.join()

    return scores


def calculate_aecmos_score(
    src_path: str,
    est_path: str = "",
    est_suffix: str = "enh",
    scenario: Dict = {
        "doubletalk": "dt",
        "nearend_singletalk": "nst",
        "farend_singletalk": "st",
    },
    sample_rate=16000,
):
    """
    est_path="" means that est wavs and source wavs are in the same directory.
    """
    if sample_rate == 16000:
        model_path = str(mos_p)
    else:
        model_path = str(mos_p_48k)

    aecmos = AECMOSEstimator(model_path)

    src_p = Path(src_path)
    item = [str(d.relative_to(src_p)) for d in src_p.iterdir() if d.is_dir()]
    if len(item) > 0:
        state_echo = _Score(*item)
        state_other = _Score(*item)
    else:
        state_echo = _Score(".")
        state_other = _Score(".")

    num = len(list(src_p.glob("**/*_lpb.wav")))

    for fname in tqdm(src_p.glob("**/*_lpb.wav"), total=num, ncols=100):
        lpb_path = str(fname)
        mic_path = lpb_path.replace("lpb", "mic")

        mic_p, mic_fname = os.path.split(mic_path)
        if est_suffix != "mic":
            mic_fname = mic_fname.replace("mic", est_suffix)
        if est_path != "":
            mic_p = mic_p.replace(src_path, est_path)

        enh_path = os.path.join(mic_p, mic_fname)

        try:
            lpb_sig, mic_sig, enh_sig = aecmos.read_and_process_audio_files(
                lpb_path, mic_path, enh_path
            )
        except Exception as e:
            print(e)
            continue

        # st, nst, dt
        sc_type = [v for k, v in scenario.items() if k in str(fname)]
        if len(sc_type) != 1:
            raise RuntimeError(f"scenario type is unclear, may be {sc_type}")
        else:
            sc_type = sc_type[0]

        scores = aecmos.run(sc_type, lpb_sig, mic_sig, enh_sig)

        item_type = str(fname.parent.relative_to(src_p))
        state_echo.update(item_type, scores[0])
        state_other.update(item_type, scores[1])

    # print(
    #     f"The AECMOS echo score is {echo_score/count:.3f}, and (other) degradation score is {other_score/count:.3f}."
    # )
    for k, v in state_echo:
        print(f"The AECMOS echo score at {k: <5} is {v: <8}")
    for k, v in state_other:
        print(f"The (other) degradation echo score at {k: <5} is {v: <8}")


# TODO
def calculate_dnsmos_score():
    pass


if __name__ == "__main__":
    calculate_aecmos_score(
        src_path="/home/ll/datasets/blind_test_set",
        est_path="/home/ll/mout/NKF",
        est_suffix="enh",
    )

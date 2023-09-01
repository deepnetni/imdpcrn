import os
import numpy as np
import soundfile as sf
import subprocess
import glob
import librosa
import random
import tempfile

EPS = np.finfo(float).eps
# np.random.seed(0)


def check_power(audio: np.ndarray, threshold=-35) -> bool:
    power = 10 * np.log10((audio**2).mean() + EPS)
    return bool(power > threshold)


def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level"""
    rms = (audio**2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    """Function to read audio"""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print("WARNING: Audio type not supported")

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio**2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms + EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0) / audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)

    return audio, sample_rate


def audiowrite(
    destpath,
    audio,
    sample_rate=16000,
    norm=False,
    target_level=-25,
    clipping_threshold=0.99,
    clip_test=False,
):
    """Function to write audio"""

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError(
                "Clipping detected in audiowrite()! "
                + destpath
                + " file not written to disk."
            )

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio / max_amp * (clipping_threshold - EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    """Return the percentage of the time the audio signal is above an energy threshold"""

    audio = normalize(audio, target_level)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + EPS)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att
            )
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel
            )

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def mix_w_snr(
    clean,
    noise,
    snr,
    target_level_lower=-35,
    target_level_upper=-15,
    target_level=-25,
    clipping_threshold=0.99,
):
    """Function to mix clean speech and noise at various SNR levels"""
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean**2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise**2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    rmsnoisy = (noisyspeech**2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        # noisy_rms_level is dBFS
        noisy_rms_level = int(
            20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS))
        )

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def mix_w_ser(speech, echo, ser, clipping_threshold=0.99):
    """Function to mix speech and echo at various SER levels
    Returns:
        speech, echo, mic, speech_scale
    """
    if len(speech) != len(echo):
        raise RuntimeError("length of speech and echo are not equal!")

    # speech = speech / (max(abs(speech)) + EPS)
    rms_speech = (speech**2).mean() ** 0.5

    # echo = echo / (max(abs(echo)) + EPS)
    rms_echo = (echo**2).mean() ** 0.5

    # Set the noise level for a given SER
    speech_scalar = rms_echo * (10 ** (ser / 20)) / (rms_speech + EPS)
    speech_new_level = speech * speech_scalar

    # Mix echo and speech
    mic_data = speech_new_level + echo

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    # TODO check if suitable
    if is_clipped(mic_data):
        mic_maxamplevel = max(abs(mic_data)) / (clipping_threshold - EPS)
        mic_data = mic_data / mic_maxamplevel
        speech_new_level = speech_new_level / mic_maxamplevel
        echo = echo / mic_maxamplevel
        speech_scalar *= 1 / mic_maxamplevel

    return speech_new_level, echo, mic_data, speech_scalar


def resampler(input_dir, target_sr=16000, ext="*.wav"):
    """Resamples the audio files in input_dir to target_sr and save"""
    files = glob.glob(f"{input_dir}/" + ext)
    for pathname in files:
        print(pathname)
        try:
            audio, fs = audioread(pathname)
            audio_resampled = librosa.core.resample(audio, fs, target_sr)
            audiowrite(pathname, audio_resampled, target_sr)
        except:
            continue


def audio_segmenter(input_dir, dest_dir, segment_len=10, ext="*.wav"):
    """Segments the audio clips in dir to segment_len in secs"""
    files = glob.glob(f"{input_dir}/" + ext)
    for i in range(len(files)):
        audio, fs = audioread(files[i])

        if len(audio) > (segment_len * fs) and len(audio) % (segment_len * fs) != 0:
            audio = np.append(
                audio, audio[0 : segment_len * fs - (len(audio) % (segment_len * fs))]
            )
        if len(audio) < (segment_len * fs):
            while len(audio) < (segment_len * fs):
                audio = np.append(audio, audio)
            audio = audio[: segment_len * fs]

        num_segments = int(len(audio) / (segment_len * fs))
        audio_segments = np.split(audio, num_segments)

        basefilename = os.path.basename(files[i])
        basename, ext = os.path.splitext(basefilename)

        for j in range(len(audio_segments)):
            newname = basename + "_" + str(j) + ext
            destpath = os.path.join(dest_dir, newname)
            audiowrite(destpath, audio_segments[j], fs)

#!/usr/bin/env python3

from __future__ import unicode_literals
import yt_dlp
import ffmpeg
import os
import shutil
import sys
import subprocess
import numpy as np
import librosa
import soundfile

Mode = "Splitting"  # @param ["Separate", "Splitting"]
dataset = "Youtube"  # @param ["Youtube", "Drive"]
url = "https://www.youtube.com/watch?v=DEqXNfs_HhY"  # @param {type:"string"}
drive_path = ""  # @param {type:"string"}
AUDIO_NAME = "test"  # @param {type:"string"}

folders = [
    f"youtubeaudio/{AUDIO_NAME}",
    f"drive/MyDrive/audio/{AUDIO_NAME}",
    f"dataset/{AUDIO_NAME}",
    f"drive/MyDrive/dataset/{AUDIO_NAME}"
]

for folder in folders:
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        pass


# Install Library for Youtube WAV Download
if dataset == "Drive":
    print("Dataset is set to Drive. Skipping this section")
elif dataset == "Youtube":
    pass

# Download Youtube WAV
if dataset == "Drive":
    print("Dataset is set to Drive. Skipping this section")
elif dataset == "Youtube":
    ydl_opts = {
        "format": "bestaudio/best",
        #    'outtmpl': 'output.%(ext)s',
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "outtmpl": f"youtubeaudio/{AUDIO_NAME}",  # this is where you can edit how you'd like the filenames to be formatted
    }

    def download_from_url(url):
        ydl.download([url])
        # stream = ffmpeg.input('output.m4a')
        # stream = ffmpeg.output(stream, 'output.wav')

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        download_from_url(url)


# Separate Vocal and Instrument/Noise using Demucs
AUDIO_INPUT = f"youtubeaudio/{AUDIO_NAME}.wav"

if dataset == "Drive":
    command = f"demucs --two-stems=vocals {drive_path}"
elif dataset == "Youtube":
    command = f"demucs --two-stems=vocals {AUDIO_INPUT}"

result = subprocess.run(command.split(), stdout=subprocess.PIPE)
print(result.stdout.decode())


# Create directory
os.makedirs(f"drive/MyDrive/audio/{AUDIO_NAME}", exist_ok=True)

# Copy files
for file in os.listdir(f"separated/htdemucs/{AUDIO_NAME}"):
    shutil.copy(
        f"separated/htdemucs/{AUDIO_NAME}/{file}", f"drive/MyDrive/audio/{AUDIO_NAME}"
    )

# Copy files if dataset is "Youtube"
if dataset == "Youtube":
    shutil.copy(f"youtubeaudio/{AUDIO_NAME}.wav", f"drive/MyDrive/audio/{AUDIO_NAME}")


# Split The Audio into Smaller Duration Before Training
if Mode == "Separate":
    print("Mode is set to Separate. Skipping this section")
elif Mode == "Splitting":
    #   !pip install numpy
    #   !pip install librosa
    #   !pip install soundfile
    os.makedirs(f"dataset/{AUDIO_NAME}", exist_ok=True)


# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            return chunks


if Mode == "Separate":
    print("Mode is set to Separate. Skipping this section")

elif Mode == "Splitting":
    audio, sr = librosa.load(
        f"separated/htdemucs/{AUDIO_NAME}/vocals.wav", sr=None, mono=False
    )  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=500,
        hop_size=10,
        max_sil_kept=500,
    )
    chunks = slicer.slice(audio)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        soundfile.write(
            f"dataset/{AUDIO_NAME}/split_{i}.wav", chunk, sr
        )  # Save sliced audio files with soundfile.

if Mode == "Separate":
    print("Mode is set to Separate. Skipping this section")
elif Mode == "Splitting":
    # Create directory
    os.makedirs(f"drive/MyDrive/dataset/{AUDIO_NAME}", exist_ok=True)

    # Copy files
    for file in os.listdir(f"dataset/{AUDIO_NAME}"):
        shutil.copy(
            f"dataset/{AUDIO_NAME}/{file}", f"drive/MyDrive/dataset/{AUDIO_NAME}"
        )

#!/usr/bin/env python3

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from __future__ import unicode_literals
import zipfile
from cog import BasePredictor, Input, Path as CogPath
from pathlib import Path
import yt_dlp
import os
import shutil
import subprocess
import numpy as np
import librosa
import soundfile


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


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        youtube_url: str = Input(
            description="URL to YouTube video you'd like to create your RVC v2 dataset from",
        ),
        audio_name: str = Input(
            default="rvc_v2_voices",
            description="Name of the dataset. The output will be a zip file containing a folder named `dataset/<audio_name>/`. This folder will include multiple `.mp3` files named as `split_<i>.mp3`. Each `split_<i>.mp3` file is a short audio clip extracted from the provided YouTube video, where voice has been isolated from the background noise.",
        ),
    ) -> CogPath:
        """Run a single prediction on the model"""

        url = youtube_url
        AUDIO_NAME = audio_name

        # Empty old folders
        folders = [
            f"youtubeaudio/{AUDIO_NAME}",
            f"drive/MyDrive/audio/{AUDIO_NAME}",
            f"dataset/{AUDIO_NAME}",
            f"drive/MyDrive/dataset/{AUDIO_NAME}",
        ]
        for folder in folders:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        # Delete old output
        test_zip = "dataset_{AUDIO_NAME}.zip"
        try:
            os.remove(test_zip)
        except FileNotFoundError:
            pass

        # Download Youtube WAV
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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Separate Vocal and Instrument/Noise using Demucs
        AUDIO_INPUT = f"youtubeaudio/{AUDIO_NAME}.wav"
        command = f"demucs --two-stems=vocals {AUDIO_INPUT}"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        print(result.stdout.decode())

        os.makedirs(f"dataset/{AUDIO_NAME}", exist_ok=True)

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

        # Correct the output ZIP file path
        output_zip_path = f"dataset_{AUDIO_NAME}.zip"

        # Zip the contents of the directory and return the CogPath of the zip file
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            base_path = Path(f"dataset/{AUDIO_NAME}")
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    file_path = Path(root) / file
                    # Define the archive path (relative path inside the ZIP, including the 'dataset/' prefix)
                    archive_path = Path("dataset") / file_path.relative_to(
                        base_path.parent
                    )
                    # Add the file to the ZIP file
                    zipf.write(file_path, archive_path.as_posix())
                    print(f"Added {file_path} as {archive_path}")

        return CogPath(output_zip_path)

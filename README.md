# RVC v2 Dataset Creation Tool

[![Replicate](https://replicate.com/zsxkib/create-rvc-dataset/badge)](https://replicate.com/zsxkib/create-rvc-dataset)

## Introduction
Create vocal datasets for Realistic Voice Cloning (RVC) v2 models with ease. Simply provide a YouTube video URL and let the tool handle the extraction and preparation of vocal data, ideal for training sophisticated voice cloning models. 🧠🎤

## Features
- **Easy Input**: Paste the URL and optionally name your dataset. 📌
- **Automated Processing**: Isolates vocals and segments them into clips. ⚙️
- **Immediate Download**: Access your dataset with a click. 💾

## How to Use
1. **YouTube URL**: Input the link to the video. 🖇️
2. **Dataset Name**: Choose a name or stick with the default. ✏️
3. **Run**: Hit 'Run' to begin the magic. 🚀
4. **Download**: Grab your `.zip` file full of vocal clips. 📦

## What You Get
- The output is a `.zip` file titled `dataset/<your_dataset_name>/`.
- It houses `split_<i>.mp3` files, each containing a clear vocal extract. 🎶

## Training Your Model
Once you have your dataset, head over to `https://replicate.com/replicate/train-rvc-model` to train your RVC model on your newly created dataset. 🚀

## Acknowledgments
This tool is adapted from work by the talented [ardha27](https://github.com/ardha27/AI-Song-Cover-RVC), who authored the initial codebase available [here](https://colab.research.google.com/github/ardha27/AI-Song-Cover-RVC/blob/main/Download_Youtube_WAV_and_Splitting_Audio.ipynb). 🙏

# Source-Filter HiFi-GAN (SiFi-GAN)

This repo provides official PyTorch implementation of [SiFi-GAN](https://arxiv.org/abs/2210.15533), a fast and pitch controllable high-fidelity neural vocoder.<br>
This repo also provides code and instruction to serve this model via Nvidia-Triton server

# How to serve model via Nvidia-Triton
## Environment setup

Install python dependencies.
```bash
cd SiFiGAN
pip install -e .
```
Install docker https://docs.docker.com/engine/install/ <br>
Install nvidia container-toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
<br>

## Prepare model for serving
### Download pretrained checkpoint.
At this example we will create new folder ```checkpoints``` at ```sifigan/nv_triton/misc``` 
and download here pretrained checkpoint.
```bash
cd sifigan/nv_triton/misc
mkdir checkpoints
# donload model from dropbox
wget -O checkpoint.pkl https://www.dropbox.com/s/w3pnnmpsxvqfykx/checkpoint-1000000steps.pkl?dl=0
```

### Prepare pretrained checkpoint for serving
You can prepare only one type of model from this section and run serving for this model.
But we will show you how to prepare all types of models

Please refer to the [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) repo for more details.

## Folder architecture
- **egs**:
The folder for projects.
- **egs/namine_ritsu**:
The folder of the [Namine Ritsu](https://www.youtube.com/watch?v=pKeo9IE_L1I) project example.
- **sifigan**:
The folder of the source codes.

The dataset preparation of Namine Ritsu database is based on [NNSVS](https://github.com/nnsvs/nnsvs/).
Please refer to it for the procedure and details.

## Run

In this repo, hyperparameters are managed using [Hydra](https://hydra.cc/docs/intro/).<br>
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation

Make dataset and scp files denoting paths to each audio files according to your own dataset (e.g., `egs/namine_ritsu/data/scp/namine_ritsu.scp`).<br>
List files denoting paths to the extracted features are automatically created in the next step (e.g., `egs/namine_ritsu/data/scp/namine_ritsu.list`).<br>
Note that scp/list files for training/validation/evaluation are needed.

### Preprocessing

```bash
# Move to the project directory
$ cd egs/namine_ritsu

# Extract acoustic features (F0, mel-cepstrum, and etc.)
# You can customize parameters according to sifigan/bin/config/extract_features.yaml
$ sifigan-extract-features audio=data/scp/namine_ritsu_all.scp

# Compute statistics of training data
$ sifigan-compute-statistics feats=data/scp/namine_ritsu_train.list stats=data/stats/namine_ritsu_train.joblib
```

### Training

```bash
# Train a model customizing the hyperparameters as you like
$ sifigan-train generator=sifigan discriminator=univnet train=sifigan data=namine_ritsu out_dir=exp/sifigan
```

### Inference

```bash
# Decode with several F0 scaling factors
$ sifigan-decode generator=sifigan data=namine_ritsu out_dir=exp/sifigan checkpoint_steps=400000 f0_factors=[0.5,1.0,2.0]
```

### Analysis-Synthesis

```bash
# WORLD analysis + Neural vocoder synthesis
$ sifigan-anasyn generator=sifigan in_dir=your_own_input_wav_dir out_dir=your_own_output_wav_dir stats=pretrained_sifigan/namine_ritsu_train_no_dev.joblib checkpoint_path=pretrained_sifigan/checkpoint-400000steps.pkl f0_factors=[1.0]
```

### Pretrained model

~~I provide a pretrained SiFiGAN model [HERE](https://www.dropbox.com/s/akofngycxxz1dg5/pretrained_sifigan.tar.gz?dl=0) which is trained on the Namine Ritsu corpus in the same training manner described in the paper.
You can download and place it in your own directory. Then set the appropriate path to the pretrained model and the command should work.~~


~~However, since the Namine Ritsu corpus includes a single female Japanese singer, there is a possibility that the model would not work well especially for male singers.
I am planning to publish another pretrained model trained on larger dataset including many speakers.~~

Due to being trained on the code before bug fixes, I have decided to cancel the release of the model trained on the Namine Ritsu database. Instead, a model trained on the following large-scale dataset is available.

A pretrained model on 24 kHz speech + singing datasets is available [HERE](https://drive.google.com/file/d/1uzqTeumvkPQpfdK_D4U41MDL5-s-hs0l/view?usp=sharing). We used train-clean-100 and train-clean-360 in [LibriTTS-R](https://google.github.io/df-conformer/librittsr/), and [NUS-48E](https://www.smcnus.org/wp-content/uploads/2013/09/05-Pub-NUS-48E.pdf) for training.
Two speakers, ADIZ and JLEE in NUS-48E, were excluded from the training data for evaluation. Also, the wav data of NUS-48E were divided into clips of approximately one second each before the feature extraction step.

The feature preprocessing and training commands are as follows:
```bash
sifigan-extract-features audio=data/scp/libritts_r_clean+nus-48e_train_no_dev.scp minf0=60 maxf0=1000
sifigan-extract-features audio=data/scp/libritts_r_clean+nus-48e_dev.scp minf0=60 maxf0=1000
sifigan-extract-features audio=data/scp/libritts_r_clean+nus-48e_eval.scp minf0=60 maxf0=1000

sifigan-compute-statistics feats=data/scp/libritts_r_clean+nus-48e_train_no_dev.list stats=data/stats/libritts_r_clean+nus-48e_train_no_dev.joblib

sifigan-train out_dir=test/sifigan generator=sifigan data=libritts_r_clean+nus-48e train=sifigan_1000k
```

### Monitor training progress

```bash
$ tensorboard --logdir exp
```

## Citation
If you find the code is helpful, please cite the following article.

```
@INPROCEEDINGS{10095298,
  author={Yoneyama, Reo and Wu, Yi-Chiao and Toda, Tomoki},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={{Source-Filter HiFi-GAN: Fast and Pitch Controllable High-Fidelity Neural Vocoder}},
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095298}
}
```

## Authors

Development:
[Reo Yoneyama](https://chomeyama.github.io/Profile/) @ Nagoya University, Japan<br>
E-mail: `yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp`

Advisors:<br>
[Yi-Chiao Wu](https://bigpon.github.io/) @ Meta Reality Labs Research, USA<br>
E-mail: `yichiaowu@fb.com`<br>
[Tomoki Toda](https://sites.google.com/site/tomokitoda/) @ Nagoya University, Japan<br>
E-mail: `tomoki@icts.nagoya-u.ac.jp`

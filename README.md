![Flowtron](https://nv-adlr.github.io/images/flowtron_logo.png "Flowtron")

## Flowtron: an Autoregressive Flow-based Network for Text-to-Mel-spectrogram Synthesis

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/flowtron.git`
2. CD into this repo: `cd flowtron`
3. Initialize submodule: `git submodule update --init; cd tacotron2; git submodule update --init`
4. Install [PyTorch]
5. Install python requirements or build docker image
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. Update the filelists inside the filelists folder to point to your data
2. `python train.py -c config.json -p train_config.output_directory=outdir`
3. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence.
Dataset dependent layers can be [ignored]

1. Download our published [Flowtron LJS] or [Flowtron LibriTTS] model
2. `python train.py -c config.json -p train_config.ignore_layers=["speaker_embedding.weight"] train_config.checkpoint_path="models/flowtron_ljs.pt"`

## Multi-GPU (distributed) and Automatic Mixed Precision Training ([AMP])
1. `python -m torch.distributed.launch --use_env --nproc_per_node=NUM_GPUS_YOU_HAVE train.py -c config.json -p train_config.output_directory=outdir train_config.fp16=true`

## Inference demo
1. `python inference.py -c config.json -f models/flowtron_ljs.pt -w models/waveglow_256channels_v4.pt -t -i 0`

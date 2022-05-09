
# EmotionBox: a music-element-driven emotional music generation system based on music psychology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of EmotionBox, the paper is available on 
[https://arxiv.org/abs/2112.08561](https://arxiv.org/abs/2112.08561).

The codes contain the proposed method and label-based method.

The trained models are in the .\save dir.

## Generated Samples

    Run the generate.py to generate music using EmotionBox.
    Run the generate_label.py to generate music using label-based method.


## Training Instructions

- Preprocessing

    ```shell
    Run preprocess.py 
    ```

- Training
    ```shell
    Run train.py 
## Requirements

- pretty_midi
- numpy
- pytorch >= 0.4
- tensorboardX
- progress

## acknowledgement

We thank for the codes of [Performance-RNN-PyTorch](https://github.com/djosix/Performance-RNN-PyTorch) from djosix.
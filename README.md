# Creative Text-to-Audio Generation via Synthesizer Programming (CTAG)
![Python version](https://img.shields.io/badge/python-3.9-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/ctag)
[![Paper](https://img.shields.io/badge/paper-NeurIPS-ML4Audio)](https://mlforaudioworkshop.com/CreativeTextToAudio.pdf)
[![Website](https://img.shields.io/badge/website-CTAG)](https://ctag.media.mit.edu/)

<a href="https://github.com/PapayaResearch/ctag/blob/main/media/logo.png"><img src="https://github.com/PapayaResearch/ctag/blob/main/media/logo.png?raw=true" width="200" align="right" /></a>

Sound designers have long harnessed the power of abstraction to distill and highlight the semantic essence of real-world auditory phenomena, akin to how simple sketches can vividly convey visual concepts. However, current neural audio synthesis methods lean heavily towards capturing acoustic realism. We introduce `ctag`, a novel method centered on meaningful abstraction. Our approach takes a text prompt and iteratively refines the parameters of a virtual modular synthesizer ([SynthAX](https://github.com/PapayaResearch/synthax)) to produce sounds with high semantic alignment, as predicted by a pretrained audio-language model.

The code to obtain the results from the paper can be found in a different [repository](https://github.com/PapayaResearch/ctag-experiments), and all the sounds are on the website!

## Installation

You can create the environment as follows

```bash
conda env create -f environment.yml
conda activate ctag
```

By default, we install JAX for CPU, and you can find more details in the [JAX documentation](https://github.com/google/jax#installation) to use JAX on your accelerators.

### Checkpoints

You also have to download the [checkpoints](https://huggingface.co/lukewys/laion_clap/tree/main) for [LAION-CLAP](https://github.com/LAION-AI/CLAP) as follows

```bash
mkdir -p ctag/checkpoints && wget -i checkpoints.txt -P ctag/checkpoints
```

## `ctag`

We use [Hydra](https://hydra.cc/) to configure `ctag`. The configuration can be found in `ctag/conf/` and `ctag/conf/config.yaml` defines all the parameters (e.g. strategy algorithm, synthesizer, iterations, prompts), which by default are the ones used for the paper. You can choose the `model` according to the downloaded CLAP `checkpoints`, a strategy available in the configuration, a `synth` architecture and a `synthconfig`. This is also where you choose the `prompts`, the `duration` of the sounds, the number of `iterations`, the `popsize`, the number of sounds generated per prompt `n_runs` (not to confuse with the iterations), and the initial random `seed`.

Generating sounds is very simple! By default, `ctag` runs on GPU, but you can change that with a flag

```bash
cd ctag
python -u text2synth.py system.device=cpu
```

It will generate directories containing logs, results, and experiments. The final version of each sound can be found in `experiments`, and `results` contains all the iterations.

### Hyperparameters

We use AX to sweep the hyperparameters of an algorithm with just a flag. First, you need to update the version of `ax-platform` because of some dependency issues with other packages

```bash
pip install ax-platform==0.2.8
```

Then you can run the sweeping as follows

```bash
python -u text2synth.py system.device=cpu --multirun
```

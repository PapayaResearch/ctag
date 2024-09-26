# Creative Text-to-Audio Generation via Synthesizer Programming (CTAG)

<a href="https://github.com/PapayaResearch/ctag/blob/main/media/logo.png"><img src="https://github.com/PapayaResearch/ctag/blob/main/media/logo.png?raw=true" width="200" align="right" /></a>

![Python version](https://img.shields.io/badge/python-3.9-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/ctag)
[![Website](https://img.shields.io/badge/website-CTAG-red)](https://ctag.media.mit.edu/)

Code for the **ICML 2024** paper **[Creative Text-to-Audio Generation via Synthesizer Programming](https://arxiv.org/abs/2406.00294)**. CTAG is a method for generating sounds from text prompts by using a virtual modular synthesizer. CTAG depends on [SynthAX](https://github.com/PapayaResearch/synthax), a fast modular synthesizer in JAX.

You can hear many examples on the [website](https://ctag.media.mit.edu/). The code to obtain the results from the paper will be found in a different [repository](https://github.com/PapayaResearch/ctag-experiments) (coming soon).

## Installation

You can create the environment as follows

```bash
conda create -n ctag python=3.9
conda activate ctag
pip install -r requirements.txt
```

By default, we install JAX for CPU. You can find more details in the [JAX documentation](https://github.com/google/jax#installation) on using JAX with your accelerators.

### CLAP Checkpoints

You also have to download the [checkpoints](https://huggingface.co/lukewys/laion_clap/tree/main) for [LAION-CLAP](https://github.com/LAION-AI/CLAP) as follows:

```bash
mkdir -p ctag/checkpoints && wget -i checkpoints.txt -P ctag/checkpoints
```

## `ctag/`
Generating sounds is very simple! By default, `ctag` runs on CPU with a lower population size, but you can change that with config values

```bash
cd ctag
python text2synth.py system.device=cuda general.popsize=100
```

It will generate directories containing logs, results, and experiments. The final version of each sound can be found in `experiments`, and `results` contains all the iterations.

By default, this uses the prompts in `ctag/data/esc50-sounds.txt`. To change this, point this field to a different file or pass a string with multiple semicolon-separated prompts. You can also override this from the command line:

```bash
# From a prompts.txt file
python text2synth.py general.prompts=/path/to/prompts.txt

# From strings
python text2synth.py general.prompts='"a bird tweeting;walking on leaves"'
```

Note that currently, you must supply $\geq$ 2 prompts! This is due to an [issue](https://github.com/LAION-AI/CLAP/pull/105) in the version of CLAP on PyPI.

## Configuration
We use [Hydra](https://hydra.cc/) to configure `ctag`. The configuration can be found in `ctag/conf/config.yaml`, with specific sub-configs in sub-directories of `ctag/conf/`.

The configs define all the parameters (e.g. strategy algorithm, synthesizer, iterations, prompts). By default, these are the ones used for the paper. You can choose the `model` according to the downloaded CLAP `checkpoints`, an `evosax` strategy available in the configuration, a `synth` architecture and a `synthconfig`. This is also where you choose the `prompts`, the `duration` of the sounds, the number of `iterations`, the `popsize` (population size), the number of independent runs per prompt `n_runs` (not to confuse with the iterations), and the initial random `seed`.

### Hyperparameters

We use AX to sweep the hyperparameters of an algorithm with just a config field. First, you need to update the version of `ax-platform` because of some dependency issues with other packages

```bash
pip install ax-platform==0.2.8
```

Then you can run the sweeping as follows

```bash
python text2synth.py --multirun
```

## Acknowledgements & Citing

If you use `ctag` in your research, please cite the following paper:
```bibtex
@inproceedings{cherep2024creative,
  title={Creative Text-to-Audio Generation via Synthesizer Programming},
  author={Cherep, Manuel and Singh, Nikhil and Shand, Jessica},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

For the synthesizer component itself, please cite [SynthAX](https://github.com/PapayaResearch/synthax):
```bibtex
@conference{cherep2023synthax,
  title = {SynthAX: A Fast Modular Synthesizer in JAX},
  author = {Cherep*, Manuel and Singh*, Nikhil},
  booktitle = {Audio Engineering Society Convention 155},
  month = {May},
  year = {2023},
  url = {http://www.aes.org/e-lib/browse.cfm?elib=22261}
}
```

We acknowledge partial financial support by Fulbright Spain. We also acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the research results reported within these papers.

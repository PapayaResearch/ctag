# Copyright (c) 2024
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import Any


#######################
# Misc. Objects
#######################

@dataclass
class Model:
    # Model name
    name: str
    # Model checkpoint (filename only)
    ckpt: str
    # Model object
    model: Any

@dataclass
class Strategy:
    # Algorithm name
    name: str
    # Algorithm object
    algorithm: Any
    # Evoparams object
    evoparams: Any

@dataclass
class SynthConfig:
    # Config name
    name: str
    # Config object
    config: Any

@dataclass
class Synth:
    # Synth name
    name: str
    # Synth object
    synth: Any

#######################
# General Settings
#######################

@dataclass
class General:
    # Path to file with text prompts
    prompts: str
    # Path to model checkpoints
    ckpt_path: str
    # Duration of output audio in seconds
    duration: float
    # Number of iterations to run the optimization for
    iter: int
    # Population size
    popsize: int

#######################
# Logging Settings
#######################

@dataclass
class Logging:
    # Whether to use Tensorboard
    use_tensorboard: bool
    # Directory for writing logs (for Tensorboard)
    log_dir: str
    # Directory for writing results (for audio and parameters)
    results_dir: str
    # Whether to write audio
    write_audio: bool
    # Whether to log parameters as embeddings (for Tensorboard)
    log_embeddings: bool
    # Log frequency (in iterations)
    logevery: int
    # Verbose logging
    verbose: bool


#######################
# Experiment Settings
#######################


@dataclass
class Experiment:
    # Directory for writing experiments
    experiment_dir: str
    # How many runs for each prompt
    n_runs: int


######################
# System Settings
######################


@dataclass
class System:
    # Random seed for reproducibility
    seed: int
    # Device to run process on. Options: ["cpu", "cuda"]
    device: str


######################
# The Config
######################

@dataclass
class Config:
    # General settings
    general: General
    # Logging settings
    logging: Logging
    # Experiment settings
    experiment: Experiment
    # System settings
    system: System
    # Model settings
    model: Model
    # Algorithm settings
    strategy: Strategy
    # Synth config
    synthconfig: SynthConfig
    # Synth settings
    synth: Synth

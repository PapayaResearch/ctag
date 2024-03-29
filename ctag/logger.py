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

import os
import logging
import numpy as np
import soundfile
import jax.numpy as jnp
import synthax.io
from synthax.synth import BaseSynth
from typing import Mapping


def verbose(
    iteration: int,
    total: int,
    value: float
):
    logging.info(
        "iter: [%d/%d], value: %f",
        iteration,
        total,
        value
    )


def log(
    synth: BaseSynth,
    params: Mapping,
    audio: jnp.ndarray,
    outdir: str,
    iteration: int,
    kind: str = "best",
    samplerate: int = 48000
):
    outf = os.path.join(outdir, "%s_%.5d" % (kind, iteration))
    audio /= jnp.max(jnp.abs(audio))

    # Save audio file
    soundfile.write(
        "%s.wav" % outf,
        np.asarray(audio).squeeze(),
        samplerate,
    )

    # Save synth specs
    synthax.io.write_synthspec(
        "%s_synthspec.yaml" % outf,
        synth,
        params,
    )

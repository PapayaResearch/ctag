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

import numpy as np
import logging


def get_hparamdict(cfg, **kwargs):
    hparamdict = {
        "model": cfg.model.name,
        "config": cfg.synthconfig.name,
        "synth": cfg.synth.name,
        "algorithm": cfg.strategy.name,
        "popsize": cfg.general.popsize,
        "iter": cfg.general.iter,
        **{k:v for k, v in cfg.strategy.algorithm.items() if v is not None},
        **kwargs
    }

    def simplify_hparams(hparams: dict):
        hparams_out = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                hparams_out[k] = v
            elif isinstance(v, (np.float16, np.float32, np.float64)):
                hparams_out[k] = float(v)
            elif isinstance(v, (np.int16, np.int32, np.int64)):
                hparams_out[k] = int(v)
            else:
                logging.warning("Could not simplify hparam %s of type %s." % (k, type(v)))
        return hparams_out

    return simplify_hparams(hparamdict)

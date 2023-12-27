# Copyright (c) 2023
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
import fcntl
from utils.softfilelock import MySoftFileLock
fcntl.flock = MySoftFileLock
import numpy as np
import pandas as pd
import torch
import jax
import flax
import evosax
import hydra
import soundfile
import synthax
import logger
from datetime import datetime
from tqdm.auto import tqdm
from jax import numpy as jnp
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from hydra.utils import instantiate
from synthax.synth import BaseSynth
from embedding import BaseModel
from config import Config
from utils.random import PRNGKey
from utils.hparam import get_hparamdict
from utils.misc import load_prompts, print_config
from typing import Callable


config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    OmegaConf.resolve(cfg)
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print_config(cfg)

    ##############################################
    # Seeding
    ##############################################
    PRNG_key = PRNGKey(cfg.system.seed)

    ##############################################
    # Load model
    ##############################################
    model = instantiate(cfg.model.model)

    ##############################################
    # Initialize synthesizer, JIT once
    ##############################################
    synth = instantiate(cfg.synth.synth)
    batch_size = cfg.synth.synth.config.batch_size
    cfg.synth.synth.config.update({"batch_size": 1})
    test_synth = instantiate(cfg.synth.synth)
    synth_apply = jax.jit(synth.apply)
    test_synth_apply = jax.jit(test_synth.apply)

    # Get initial parameters to initialize algorithm
    params = synth.init(PRNG_key.split())
    fparams_dict = flax.traverse_util.flatten_dict(params)
    fparams_array = jnp.concatenate(
        [x.reshape(batch_size, -1) for x in fparams_dict.values()],
        axis=1
    )
    init_paramsdict = test_synth.init(PRNG_key.split())
    finit_paramsdict = flax.traverse_util.flatten_dict(init_paramsdict)
    init_params = flax.traverse_util.unflatten_dict(dict(zip(
        finit_paramsdict.keys(),
        [x.squeeze() for x in finit_paramsdict.values()]
    )))

    reshaper = evosax.ParameterReshaper(init_params)
    reshape = jax.jit(reshaper.reshape)

    ##############################################
    # Load prompts
    ##############################################
    prompts = load_prompts(cfg.general.prompts)

    ##############################################
    # Calculate text embeddings once
    ##############################################
    text_embeddings = model.embed_text(prompts)

    ##############################################
    # Iterate over prompts
    ##############################################
    current_datetime = datetime.now().strftime("%a-%b-%d-%Y_%I-%M%p")

    pbar = tqdm(prompts)
    best_fitnesses = []
    for i, prompt in enumerate(pbar):
        pbar.set_postfix({"prompt": prompt})

        pbar_runs = tqdm(range(cfg.experiment.n_runs), leave=False)
        for run in pbar_runs:
            pbar_runs.set_postfix({"run": run})

            ##############################################
            # Reinitialize synth per run
            ##############################################
            params = synth.init(PRNG_key.split())
            fparams_dict = flax.traverse_util.flatten_dict(params)
            fparams_array = jnp.concatenate(
                [x.reshape(batch_size, -1) for x in fparams_dict.values()],
                axis=1
            )

            ##############################################
            # Create the logs and results directories
            ##############################################
            base_dir = os.path.join(
                cfg.strategy.name,
                prompt,
                cfg.model.name,
                cfg.synthconfig.name,
                cfg.synth.name,
                "run-%d_%s" % (run, current_datetime)
            )

            log_dir = os.path.join(cfg.logging.log_dir, base_dir)
            results_dir = os.path.join(cfg.logging.results_dir, base_dir)
            experiment_dir = os.path.join(
                cfg.experiment.experiment_dir,
                current_datetime,
                os.path.splitext(os.path.basename(cfg.general.prompts))[0]
            )

            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(experiment_dir, exist_ok=True)

            with open(os.path.join(results_dir, "cfg.yaml"), "w") as outfile:
                outfile.write(cfg_yaml)

            writer = SummaryWriter(log_dir) if cfg.logging.use_tensorboard else None

            ##############################################
            # Run optimization
            ##############################################

            # Initialize strategy
            strategy = instantiate(
                cfg.strategy.algorithm,
                popsize=cfg.general.popsize,
                num_dims=reshaper.total_params
            )

            es_params = strategy.default_params
            es_params = es_params.replace(**dict(cfg.strategy.evoparams.items()))
            state = strategy.initialize(
                PRNG_key.split(),
                es_params,
                fparams_array
            )

            history = []

            # Optimization loop
            pbar_inner = tqdm(range(cfg.general.iter), leave=False)
            for t in pbar_inner:
                ##############################################
                # Generate batch of parameters
                ##############################################
                synth_params_array, state = strategy.ask(PRNG_key.split(), state, es_params)

                ##############################################
                # Synthesize sound and evaluate the similarity
                ##############################################
                synth_params_repaired = reshape(synth_params_array)

                fitness = make_sound_from_text(
                    synth_params_repaired,
                    synth_apply=synth_apply,
                    synth=synth,
                    prompt=text_embeddings[i],
                    model=model
                )

                state = strategy.tell(synth_params_array, fitness, state, es_params)

                # Save best fitness for logs and history
                best_fitness = float(state.best_fitness)
                pbar_inner.set_postfix({"fitness": best_fitness})
                history.append({"best": best_fitness})

                if cfg.logging.verbose:
                    logger.verbose(
                        t,
                        cfg.general.iter,
                        best_fitness
                    )

                ##############################################
                # Produce audio for best and center parameters
                ##############################################
                best_member_params = reshape(
                    jnp.expand_dims(
                        state.best_member,
                        axis=0
                    )
                )
                audio_best = test_synth_apply(best_member_params)

                center_member = state.mean[
                    jnp.linalg.norm(
                        state.mean - state.mean.mean(axis=0),
                        axis=1
                    ).argmin()
                ]
                center_member_params = reshape(
                    jnp.expand_dims(
                        center_member,
                        axis=0
                    )
                )
                audio_center = test_synth_apply(center_member_params)

                ##############################################
                # Log audio and synthesis parameters
                ##############################################
                if cfg.logging.write_audio and (t % cfg.logging.logevery == 0):
                    logger.log(
                        test_synth,
                        best_member_params,
                        audio_best,
                        results_dir,
                        t,
                        kind="best"
                    )
                    logger.log(
                        test_synth,
                        center_member_params,
                        audio_center,
                        results_dir,
                        t,
                        kind="center"
                    )

                ##############################################
                # Log scalars and audio TensorBoard
                ##############################################
                if cfg.logging.use_tensorboard and (t % cfg.logging.logevery == 0):
                    if cfg.logging.write_audio:
                        writer.add_audio(
                            "audio/best",
                            np.asarray(audio_best.squeeze()),
                            global_step=t,
                            sample_rate=int(synth.sample_rate),
                        )
                        writer.add_audio(
                            "audio/center",
                            np.asarray(audio_center.squeeze()),
                            global_step=t,
                            sample_rate=int(synth.sample_rate),
                        )

            hparamdict = get_hparamdict(
                cfg,
                prompt=prompt,
                **es_params.__dict__
            )

            if cfg.logging.use_tensorboard:
                writer.add_hparams(hparamdict, {"hparam/fitness": best_fitness})
                writer.flush()

            ##############################################
            # Write audio for experiment
            ##############################################
            audio_outpath = os.path.join(
                experiment_dir,
                "%s_%.3d_%.4f.wav" % (prompt, run, best_fitness)
            )

            params_outpath = os.path.join(
                experiment_dir,
                "%s_%.3d_%.4f" % (prompt, run, best_fitness)
            )

            for k, v in state.__dict__.items():
                v = np.asarray(v)
                if isinstance(v, (type(jax.typing.ArrayLike), str)) and v.ndim == 0:
                    writer.add_scalar(k, v, global_step=t)
                elif cfg.logging.log_embeddings:
                    writer.add_embedding(np.atleast_2d(v), tag=k, global_step=t)

            soundfile.write(
                audio_outpath,
                np.asarray(audio_best.squeeze()),
                int(synth.sample_rate)
            )

            synthax.io.write_synthspec(
                params_outpath,
                test_synth,
                best_member_params
            )

            best_fitnesses.append(best_fitness)

        ##############################################
        # Write history to csv
        ##############################################
        pd.DataFrame(history).to_csv(os.path.join(results_dir, "history.csv"), index=False)

    return sum(best_fitnesses)/len(best_fitnesses)



@torch.inference_mode()
def make_sound_from_text(
        synth_params_repaired: flax.core.frozen_dict.FrozenDict,
        synth_apply: Callable,
        synth: BaseSynth,
        prompt: torch.Tensor,
        model: BaseModel
):
    """Synthesize sound from text and calculate the similarity with the prompt.

    Args:
        synth_params_repaired (flax.core.frozen_dict.FrozenDict): Synthesizer parameters.
        synth_apply: Callable: The (potentially JIT compiled) synthesis function.
        synth (BaseSynth): Synthesizer (subclass of AbstractSynth).
        prompt (torch.Tensor): Prompt (as embedding vector).
        model (BaseModel): Model to embed audio (currently should be CLAPModule or ImageBindModel).
    """
    # Generate audio for each
    audios = synth_apply(synth_params_repaired)
    audios_torch = torch.from_numpy(np.asarray(audios)).to(model.device)

    # Calculate all audio embeddings
    prompt = torch.atleast_2d(prompt)
    audios = torch.atleast_2d(audios_torch)

    audio_embeddings = model.embed_audio(audios, int(synth.sample_rate))

    # Calculate the similarity with the prompt
    similarity = audio_embeddings @ prompt.T
    similarity_jax = jnp.asarray(similarity.cpu())

    # Return the similarity score, as evosax minimizes the objective function
    return -similarity_jax.squeeze()

if __name__ == "__main__":
    main()

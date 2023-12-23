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

import abc
import torch
import laion_clap
from typing import Union, Iterable


class BaseModel(abc.ABC):
    def embed_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Embeds audio into a latent space.

        Args:
            audio: Tensor of shape (batch_size, n_samples)

        Returns:
            Tensor of shape (batch_size, n_features)
        """
        raise NotImplementedError("embed_audio not implemented for this model.")

    def embed_text(self, text: Union[str, Iterable[str]]) -> torch.Tensor:
        """
        Embeds text into a latent space.

        Args:
            text: A string or list of strings

        Returns:
            Tensor of shape (batch_size, n_features)
        """
        raise NotImplementedError("embed_text not implemented for this model.")


class CLAPModel(BaseModel):
    def __init__(
        self,
        ckpt_path: str,
        enable_fusion: bool,
        amodel: str,
        tmodel: str,
        compile: bool = True,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion,
            amodel=amodel,
            tmodel=tmodel,
            device=device
        )
        self.compile = compile
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.model.model.load_state_dict(state_dict)
        if self.compile:
            self.model = torch.compile(self.model)

    def embed_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.model.get_audio_embedding_from_data(audio, use_tensor=True)

    def embed_text(self, text: Union[str, Iterable[str]]) -> torch.Tensor:
        return self.model.get_text_embedding(text, use_tensor=True)

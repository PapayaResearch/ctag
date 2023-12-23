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
import rich
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.box import ROUNDED
from omegaconf import OmegaConf, DictConfig


def load_prompts(prompts_file: str):
    if os.path.isfile(prompts_file):
        with open(prompts_file) as infile:
            prompts = [s.strip() for s in infile.read().split("\n")][:-1]
    else:
        prompts = [prompt.strip() for prompt in prompts_file.split(";")]
    return prompts


def print_config(cfg: DictConfig):
    table = Table(title="Hydra Configuration", box=ROUNDED)

    # Styling options
    table.row_styles = [Style(color="cyan", dim=True), Style(color="magenta", dim=True)]  # Zebra style

    # Collect all keys
    keys = set()
    for key, value in cfg.items():
        keys.add(key)
        if isinstance(value, dict):
            for nested_key in value.keys():
                keys.add(nested_key)

    # Add columns for each key
    for key in keys:
        table.add_column(key, style="cyan")

    # Populate values in respective columns
    row = {}
    for key, value in cfg.items():
        if isinstance(value, dict): # What we want here is to recursively resolve into tables
            for nested_key, nested_value in value.items():
                row[nested_key] = OmegaConf.to_yaml(nested_value)
        else:
            row[key] = OmegaConf.to_yaml(value)

    # Add a row with values in respective columns
    table.add_row(*[row.get(key, "") for key in keys])

    console = Console()
    console.print(table)

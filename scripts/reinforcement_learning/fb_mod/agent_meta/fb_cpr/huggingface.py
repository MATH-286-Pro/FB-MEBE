# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from huggingface_hub import PyTorchModelHubMixin
from .model import FBcprModel as BaseFBcprModel

class FBcprModel(
    BaseFBcprModel,
    PyTorchModelHubMixin,
    library_name="metamotivo",
    tags=["facebook", "meta", "pytorch"],
    license="cc-by-nc-4.0",
    repo_url="https://github.com/facebookresearch/metamotivo",
    docs_url="https://metamotivo.metademolab.com/",
): ...

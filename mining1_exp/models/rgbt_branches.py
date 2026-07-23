from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .yolo_adapter import ModelContractError, YoloV8nAdapter


T1_MODALITIES = {"visible_only", "thermal_only"}
THERMAL_SINGLE_CHANNEL_TRANSFORM = "replicate_3ch"


@dataclass(frozen=True)
class BranchBatch:
    images: torch.Tensor
    record_ids: tuple[str, ...]
    modality: str

    def model_tensor(self) -> torch.Tensor:
        return self.images


def _prepare_one(image: np.ndarray, modality: str) -> torch.Tensor:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ModelContractError("branch input must be a non-empty uint8 NumPy image")
    if modality == "visible_only":
        if image.ndim != 3 or image.shape[2] != 3:
            raise ModelContractError("visible input must be HWC with three channels")
        prepared = image
    elif modality == "thermal_only":
        if image.ndim == 2:
            prepared = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            prepared = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 3:
            prepared = image
        else:
            raise ModelContractError("thermal input must be HW, HWC1, or HWC3")
    else:
        raise ModelContractError(f"unsupported T1 modality: {modality}")
    tensor = torch.from_numpy(np.ascontiguousarray(prepared)).permute(2, 0, 1)
    return tensor.to(dtype=torch.float32).div_(255.0)


def build_branch_batch(
    images: Sequence[np.ndarray],
    *,
    record_ids: Iterable[str],
    modality: str,
) -> BranchBatch:
    if modality not in T1_MODALITIES:
        raise ModelContractError(f"unsupported T1 modality: {modality}")
    identifiers = tuple(str(value) for value in record_ids)
    if len(images) != len(identifiers) or not images:
        raise ModelContractError("images and record IDs must be non-empty and aligned")
    if any(not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ModelContractError("record IDs must be unique non-empty audit keys")
    tensors = [_prepare_one(image, modality) for image in images]
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ModelContractError("branch batch images must share one tensor shape")
    return BranchBatch(images=torch.stack(tensors), record_ids=identifiers, modality=modality)


class IndependentT1Branches(nn.Module):
    def __init__(
        self,
        visible: YoloV8nAdapter,
        thermal: YoloV8nAdapter,
        *,
        training_seed: int = 1701,
    ) -> None:
        super().__init__()
        if visible is thermal or visible.backend is thermal.backend:
            raise ModelContractError("T1 visible and thermal branches must be independent")
        visible_parameters = {id(parameter) for parameter in visible.parameters()}
        thermal_parameters = {id(parameter) for parameter in thermal.parameters()}
        if visible_parameters.intersection(thermal_parameters):
            raise ModelContractError("T1 branches may not share trainable parameters")
        self.visible = visible
        self.thermal = thermal
        self.training_seed = int(training_seed)

    def forward_visible(self, batch: BranchBatch):
        if batch.modality != "visible_only":
            raise ModelContractError("visible branch received a non-visible batch")
        return self.visible(batch.model_tensor())

    def forward_thermal(self, batch: BranchBatch):
        if batch.modality != "thermal_only":
            raise ModelContractError("thermal branch received a non-thermal batch")
        return self.thermal(batch.model_tensor())

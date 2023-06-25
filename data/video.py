import imageio
import requests
import torch
import torchvision
import zipfile
import random

from pathlib import Path
from typing import Any, Callable, Optional

class UCF101(torchvision.datasets.UCF101):
    """CIFAR10 dataset without labels."""

    def __init__(self, *args, patch_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_path = "/content/drive/MyDrive/DLLab/datasets/UCF101/ucfTrainTestlist"
        self.patch_shape = patch_shape
        self.random_crop = patch_shape != -1


    def __getitem__(self, index):
        video, audio, info, video_idx = self.video_clips.get_clip(index)
        #label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        if self.random_crop:
            video = random_crop3d(video, self.patch_shape)
            

        return video


def random_crop3d(data, patch_shape):
    if not (
        0 < patch_shape[0] <= data.shape[-3]
        and 0 < patch_shape[1] <= data.shape[-2]
        and 0 < patch_shape[2] <= data.shape[-1]
    ):
        print(data.shape)
        print(patch_shape)
        raise ValueError("Invalid shapes.")
    depth_from = random.randint(0, data.shape[-3] - patch_shape[0])
    height_from = random.randint(0, data.shape[-2] - patch_shape[1])
    width_from = random.randint(0, data.shape[-1] - patch_shape[2])
    return data[
        ...,
        depth_from : depth_from + patch_shape[0],
        height_from : height_from + patch_shape[1],
        width_from : width_from + patch_shape[2],
    ]
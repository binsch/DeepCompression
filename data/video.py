import imageio
import requests
import torch
import torchvision
import zipfile

from pathlib import Path
from typing import Any, Callable, Optional

class UCF101(torchvision.datasets.UCF101):
    """CIFAR10 dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_path = "/content/drive/MyDrive/DLLab/datasets/UCF101/ucfTrainTestlist"


    def __getitem__(self, index):
        video, audio, info, video_idx = self.video_clips.get_clip(index)
        #label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video
import torch
from torch.utils.data import Dataset

import torchvision

from bisect import bisect

import coinpp.conversion as conversion
import data.image as image
from helpers import get_datasets_and_converter, get_dataset_root

class ModulationDataset(Dataset):
    def __init__(self, run_id, filename, device, normalize=True, dataset_name=None):
        self.modulations = self.load_modulations(run_id, filename, device)
        self.is_patching_enabled = (type(self.modulations) is list)
        if self.is_patching_enabled:
            self.latent_dim = self.modulations[0].shape[1]
            self.num_samples = len(self.modulations)
            self.num_patches_per_sample = [x.shape[0] for x in self.modulations]
            self.sample_starts = [0]
            for n in self.num_patches_per_sample:
                self.sample_starts.append(self.sample_starts[-1] + n)

            self.modulations = torch.concatenate(self.modulations)
            self.length = self.modulations.shape[0]
        else:
            self.num_samples, self.latent_dim = self.modulations.shape
            self.length = self.num_samples
        
        self.normalize = normalize
        if normalize:
            self.mean = torch.mean(self.modulations, dim=0)
            self.std = torch.std(self.modulations, dim=0)
        
        self.return_original = (dataset_name is not None)

        if not self.return_original:
            return
        
        if dataset_name == "cifar10":
            self.converter = conversion.Converter("image")

            transforms = [torchvision.transforms.ToTensor()]
            self.original_dataset = image.CIFAR10(
                root=get_dataset_root("cifar10"),
                train=True,
                transform=torchvision.transforms.Compose(transforms),
                download=True,
            )


    def load_modulations(self, run_id, filename, device):
        # Load modulations tensor
        if device == torch.device("cpu"):
            modulations = torch.load(
                f"wandb/{run_id}/{filename}", map_location=torch.device("cpu")
            )
        else:
            modulations = torch.load(f"wandb/{run_id}/{filename}")
        # If modulations is a list, we are using patching. Iterate over every
        # element of list and transfer each tensor to correct device
        if type(modulations) is list:
            modulations = [mods.to(device) for mods in modulations]
        else:
            modulations = modulations.to(device)
        return modulations


    def get_sample_index(self, idx):
        if self.is_patching_enabled:
            return bisect(self.sample_starts, idx) -  1
        else:
            return idx


    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        if self.return_original:
            original = self.original_dataset[idx]
        else:
            original = None
        if self.normalize:
            return (self.modulations[idx] - self.mean) / self.std, original
        else:
            return self.modulations[idx], original


if __name__ == "__main__":
    run_id = "9p36fasn"
    filename = "modulations_train_3_steps.pt"
    device = "cuda"

    batch_size = 32

    dataset = ModulationDataset(run_id, filename, device)
    print(dataset[0])
    print(len(dataset))

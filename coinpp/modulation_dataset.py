import torch
from torch.utils.data import Dataset

class ModulationDataset(Dataset):
    def __init__(self, run_id, filename, device):
        self.modulations = self.load_modulations(run_id, filename, device)
        self.is_patching_enabled = (type(self.modulations is list))
        if self.is_patching_enabled:
            self.num_patches, self.latent_dim = self.modulations[0].shape
            self.num_samples = len(self.modulations)
            self.length = self.num_samples * self.num_patches
        else:
            self.num_samples, self.latent_dim = self.modulations.shape
            self.length = self.num_samples


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


    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        if self.is_patching_enabled:
            sample_idx = idx // (self.num_patches * self.latent_dim)
            patch_idx = (idx % (self.num_patches * self.latent_dim)) // self.latent_dim
            return self.modulations[sample_idx][patch_idx]
        else:
            return self.modulations[idx]


if __name__ == "__main__":
    run_id = "9p36fasn"
    filename = "modulations_train_3_steps.pt"
    device = "cpu"

    batch_size = 32

    dataset = ModulationDataset(run_id, filename, device)

from coinpp.modulation_dataset import ModulationDataset
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from coinpp.models import AnalysisTransform, SynthesisTransform
import numpy as np
import wandb
from itertools import chain

class QuantizationTrainer:
    def __init__(
        self,
        analysis_tranform,
        synthesis_transform,
        modulation_dataset,
        batch_size=32,
        validation_split=0.2,
        lr=1e-4,
        model_path="",
        device="cuda:0",
        use_wandb=False
    ):
        """Module to handle quantization learning.

        Args:
            func_rep (models.ModulatedSiren):
            converter (conversion.Converter):
            args: Training arguments (see main.py).
            train_dataset:
            test_dataset:
            patcher: If not None, patcher that is used to create random patches during
                training and to partition data into patches during validation.
            model_path: If not empty, wandb path where best (validation) model
                will be saved.
        """
        self.analysis_transform = analysis_tranform
        self.synthesis_transform = synthesis_transform

        self.optimizer = torch.optim.Adam(
            chain(self.analysis_transform.parameters(), self.synthesis_transform.parameters()),
            lr=lr
        )

        self.train_dataloader, self.val_dataloader = self._process_datasets(modulation_dataset, batch_size, validation_split)

        self.model_path = model_path
        self.step = 0
        self.best_val_loss = 999999
        self.device = device
        self.use_wandb = use_wandb


    def _process_datasets(self, dataset, batch_size, validation_split, shuffle_dataset=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(validation_split * dataset_size)
        if shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        return train_dataloader, val_dataloader

    def train_epoch(self):
        """Train model for a single epoch."""
        for data in self.train_dataloader:
            self.optimizer.zero_grad()
            data = data.to(self.device)

            entropy_codes = self.analysis_transform(data)
            reconstructions = self.synthesis_transform(entropy_codes)

            # Update parameters of the networks
            loss = mse_loss(data, reconstructions)
            loss.backward()
            self.optimizer.step()

            self.step += 1

            torch.set_printoptions(precision=25)
            print(
                f'Step {self.step}, Loss {loss:.8f}'
            )

            if self.use_wandb:
                wandb.log({'loss': loss}, step=self.step)
        self.validation()

    def validation(self):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Step {self.step}:")

        # Initialize validation logging dict
        log_dict = {}

        # Evaluate model for different numbers of inner loop steps
        log_dict["val_loss"] = 0.0

        # Fit modulations for each validation datapoint
        for i, data in enumerate(self.val_dataloader):
            data = data.to(self.device)

            entropy_codes = self.analysis_transform(data)
            reconstructions = self.synthesis_transform(entropy_codes)
            loss = mse_loss(data, reconstructions)

            log_dict[f"val_loss"] += loss.item()

        # Calculate average loss by dividing by number of batches
        log_dict[f"val_loss"] /= i + 1
    
        print(
            f"Loss {log_dict['val_loss']:.8f}"
        )

        if log_dict[f"val_loss"] > self.best_val_loss:
            self.best_val_loss = log_dict[f"val_loss"]
            # Optionally save new best model
            if self.use_wandb and self.model_path:
                torch.save(
                    {
                        "analysis_state_dict": self.analysis_transform.state_dict(),
                        "synthesis_state_dict": self.synthesis_transform.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    },
                    self.model_path,
                )

        if self.use_wandb:
            wandb.log(log_dict, step=self.step)

        print("\n")

if __name__ == "__main__":
    run_id = "9p36fasn"
    filename = "modulations_train_3_steps.pt"
    device = "cuda:0"

    latent_dim = 1024
    encoding_dim = 512
    hidden_dim = 1024
    num_res_blocks = 2
    use_batch_norm = True

    batch_size = 4096
    epochs = 10

    dataset = ModulationDataset(run_id, filename, device)

    analysis_transform = AnalysisTransform(
        latent_dim,
        encoding_dim,
        hidden_dim,
        num_res_blocks,
        use_batch_norm=use_batch_norm
    ).to(device)

    synthesis_transform = SynthesisTransform(
        encoding_dim,
        latent_dim,
        hidden_dim,
        num_res_blocks,
        use_batch_norm=use_batch_norm
    ).to(device)


    quantization_trainer = QuantizationTrainer(
        analysis_transform,
        synthesis_transform,
        dataset,
        device=device,
        batch_size=batch_size
    )

    for i in range(epochs):
        quantization_trainer.train_epoch()
    
    quantization_trainer.validation()

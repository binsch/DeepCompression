# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from compressai.datasets import ImageFolder
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

from coinpp.models import FactorizedPrior
from coinpp.modulation_dataset import ModulationDataset
from coinpp.rate_distortion import RateDistortionLoss
from coinpp.losses import mse2psnr
from wandb_utils import load_model


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, reconstruction_model, converter
):
    model.train()
    reconstruction_model.train()
    device = next(model.parameters()).device

    mean, std = train_dataloader.dataset.mean.to(device), train_dataloader.dataset.std.to(device)

    for i, (modulations, originals) in enumerate(train_dataloader):
        modulations = modulations.to(device)
        if originals is not None:
            originals = originals.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(modulations)
        out_net["x_hat"] = out_net["x_hat"] * std + mean

        coordinates, _ = converter.to_coordinates_and_features(originals)

        reconstructions = reconstruction_model.modulated_forward(coordinates, out_net["x_hat"])
        reconstructions = converter.to_data(coordinates, reconstructions)

        out_criterion = criterion(out_net, modulations, reconstructions, originals)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(modulations)}/{modulations.shape[0]*len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f} |"
                f'\tPSNR: {mse2psnr(out_criterion["mse_loss"]).mean().item()}'
            )


def test_epoch(epoch, test_dataloader, model, criterion, reconstruction_model, converter):
    model.eval()
    reconstruction_model.eval()
    device = next(model.parameters()).device

    mean = test_dataloader.dataset.mean.to(device)
    std = test_dataloader.dataset.std.to(device)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for (modulations, originals) in test_dataloader:
            # for some reason the lower bound is set to 0 here, so we set it to 1e-9 manually...
            model.entropy_bottleneck.likelihood_lower_bound.bound = torch.tensor([1e-9], device=device)
            modulations = modulations.to(device)
            if originals is not None:
                originals = originals.to(device)
            out_net = model(modulations)
            out_net["x_hat"] = out_net["x_hat"] * std + mean

            coordinates, features = converter.to_coordinates_and_features(originals)

            reconstructions = reconstruction_model.modulated_forward(coordinates, out_net["x_hat"])
            reconstructions = converter.to_data(coordinates, reconstructions)

            out_criterion = criterion(out_net, modulations, reconstructions, originals)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

        original = torch.permute(originals[0].cpu().detach(), (1, 2, 0))
        reconstruction = torch.permute(reconstructions[0].cpu().detach(), (1, 2, 0))
        imgs = torch.concatenate((original, reconstruction))
        plt.imshow(imgs)
        plt.show()

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def process_datasets(dataset, batch_size, test_split, shuffle_dataset=True, num_workers=0, pin_memory=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(test_split * dataset_size)
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, test_dataloader


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--wandb-run-path", type=str, help="Path of wandb run")
    parser.add_argument("--filename", type=str, help="Filename of .pt file containing the parameters of the modulation net")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=2048,
        help="Input dim of analysis tranform, i.e. size of the modulations",
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=2048,
        help="Dim of the encoding, i.e. dim of output of analysis transform and input of synthesis transform.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Dim of the hidden layers of the analysis and synthesis transforms.",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=2,
        help="Number of res blocks in the analysis and synthesis transforms.",
    )
    parser.add_argument(
        "--use-batch-norm",
        type=int,
        default=0,
        help="Whether to use batch norm in res block of analysis and synthesis transforms.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="selu",
        help="Name of the activation function to use. Options: selu, leaky",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    print("preprocessing data")
    run_id = args.wandb_run_path.split("/")[-1]
    dataset = ModulationDataset(run_id, args.filename, "cpu", dataset_name="cifar10") # should always be on CPU because it's too big

    train_dataloader, test_dataloader = process_datasets(dataset, args.batch_size, 0.2, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    net = FactorizedPrior(args.input_dim, args.encoding_dim, args.hidden_dim, args.num_res_blocks, use_batch_norm=args.use_batch_norm, activation=args.activation)
    net = net.to(device)

    print("loading reconstruction model")

    reconstruction_model, reconstruction_model_args, patcher = load_model(args.wandb_run_path, device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            reconstruction_model,
            dataset.converter
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, reconstruction_model, dataset.converter)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )


if __name__ == "__main__":
    main(sys.argv[1:])

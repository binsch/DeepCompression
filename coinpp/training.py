import coinpp.conversion as conversion
import coinpp.losses as losses
import coinpp.metalearning as metalearning
import torch
import wandb


class Trainer:
    def __init__(
        self,
        func_rep,
        converter,
        args,
        train_dataset,
        test_dataset,
        patcher=None,
        model_path="",
        model_train_path="",
        checkpoint=None
    ):
        """Module to handle meta-learning of COIN++ model.

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
        self.func_rep = func_rep
        self.converter = converter
        self.args = args
        self.patcher = patcher

        self.outer_optimizer = torch.optim.Adam(
            self.func_rep.parameters(), lr=args.outer_lr
        )

        #CONTINUED TRAINING
        #self.outer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        self.train_dataset = train_dataset[0]
        self.test_dataset = test_dataset[0]
        self.train_dataset_cif = train_dataset[1]
        self.test_dataset_cif = test_dataset[1]
        self.train_dataset_era5 = train_dataset[2]
        self.test_dataset_era5 = test_dataset[2]
        self._process_datasets()

        self.model_path = model_path
        self.step = checkpoint["step"]+1
        self.best_val_psnr = 0.0
        self.last_val_psnrs = [0.0,0.0]
        self.stop_modality = [0,0]
        self.epoch = 0

        self.modalities = ["audio", "image", "manifold"]
        self.prev_loss = [1,1,1]
        self.logs = [{"a":1},{"d":2},{"d":2}]



    def _process_datasets(self):
        """Create dataloaders for datasets based on self.args."""

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        
        self.train_dataloader_cif = torch.utils.data.DataLoader(
            self.train_dataset_cif,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        self.test_dataloader_cif = torch.utils.data.DataLoader(
            self.test_dataset_cif,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        
        self.train_dataloader_era5 = torch.utils.data.DataLoader(
            self.train_dataset_era5,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        self.test_dataloader_era5 = torch.utils.data.DataLoader(
            self.test_dataset_era5,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        
        

    def train_epoch(self, epoch):
        """Train model for a single epoch."""
        self.epoch = epoch
        audio_iter = iter(self.train_dataloader)
        image_iter = iter(self.train_dataloader_cif)
        manifold_iter = iter(self.train_dataloader_era5)

        iters = [audio_iter, image_iter, manifold_iter]
        

        for index in range(len(self.train_dataloader_cif)-2):
          #self.step > 3000 and self.stop_modality[0] == 0
          '''
          TRANSFERENCE CODE
          data_audio = next(audio_iter).to(self.args.device)
          coordinates_audio, features_audio = self.converter[0].to_coordinates_and_features(data_audio)
          
          data_image = next(image_iter).to(self.args.device)
          coordinates_image, features_image = self.converter[1].to_coordinates_and_features(data_image)
          
          #data_manifold = next(manifold_iter).to(self.args.device)
          #coordinates_manifold, features_manifold = self.converter[2].to_coordinates_and_features(data_manifold)


          data_manifold = None
          coordinates_manifold = None
          features_manifold = None
          

          data = [data_audio,data_image,data_manifold]
          coordinates = [coordinates_audio,coordinates_image,coordinates_manifold]
          features = [features_audio,features_image,features_manifold]
          
          audio_transference = [[]]
          image_transference = [[]]
          manifold_transference = [[]]

          transference = [audio_transference, image_transference, manifold_transference]
          
          if self.step % 500 == 0 and self.step != 0:
            torch.save(
                        {
                            "args": self.args,
                            "model_state_dict": self.func_rep.state_dict(),
                            'optimizer_state_dict': self.outer_optimizer.state_dict(),
                            'step': self.step,
                            'epoch': self.epoch,
                        },
                        "/content/drive/MyDrive/DLLab/mac_dll/checkpnt.pt",
                    )
          '''
          
          for modality in [0,1]:

            if True:
                if index > 443 and modality == 0:
                  continue

                data = next(iters[modality]).to(self.args.device)
                coordinates, features = self.converter[modality].to_coordinates_and_features(data)
                #print("before outputs")
                outputs = metalearning.outer_step(
                    self.func_rep,
                    coordinates,
                    features,
                    inner_steps=self.args.inner_steps,
                    inner_lr=self.args.inner_lr,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=self.args.gradient_checkpointing,
                    modality=modality
                )
                #print("after outputs")
                
                # Update parameters of base network
                self.outer_optimizer.zero_grad()
                outputs["loss"].backward(create_graph=False)
                self.outer_optimizer.step()
                '''
                MORE TRANSFERENCE CODE
                self.prev_loss[modality] = outputs["loss"]

                count = 0
                for i in [0,1]:
                  #print("before outputs_i")
                  outputs_i = metalearning.outer_step(
                      self.func_rep,
                      coordinates[i],
                      features[i],
                      inner_steps=self.args.inner_steps,
                      inner_lr=self.args.inner_lr,
                      is_train=False,
                      return_reconstructions=False,
                      gradient_checkpointing=self.args.gradient_checkpointing,
                      modality=i
                  )
                  #print("after outputs_i")

                  #transference[modality][i] = outputs_i["loss"]/prev_loss[i]
                  self.logs[count] = {f"{self.modalities[modality]}->{self.modalities[i]}": outputs_i["loss"]/self.prev_loss[i]}
                  
                  count = count+1
                '''
                  
                


                if self.step % self.args.validate_every == 0 and self.step != 0:
                    self.validation(1)
                    # I needed to validate audio separately
                    #self.validation(0)


                log_dict = {"loss": outputs["loss"].item(), "psnr": outputs["psnr"]}


  

                self.step += 1

                torch.set_printoptions(precision=25)
                print(
                    f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f}'
                )
                print(
                    f'min_inner_lr_audio {torch.min(self.func_rep.inner_lr[0][2])}, max_inner_lr_audio {torch.max(self.func_rep.inner_lr[0][2])}'
                )
                print(
                    f'min_inner_lr_image {torch.min(self.func_rep.inner_lr[1][0])}, max_inner_lr_image {torch.max(self.func_rep.inner_lr[1][0])}'
                )
                print(
                    f'min_audio_modulations {torch.min(self.func_rep.modality_modulations[0])}, max_audio_modulations {torch.max(self.func_rep.modality_modulations[0])}'
                )

                print(
                    f'min_image_modulations {torch.min(self.func_rep.modality_modulations[1])}, max_image_modulations {torch.max(self.func_rep.modality_modulations[1])}'
                )

                print(
                    f'min_manifold_modulations {torch.min(self.func_rep.modality_modulations[2])}, max_manifold_modulations {torch.max(self.func_rep.modality_modulations[2])}'
                )

                if self.args.use_wandb:
                    wandb.log(log_dict, step=self.step)
                    #TRANSFERENCE LOGS
                    #wandb.log(self.logs[0], step =self.step)
                    #wandb.log(self.logs[1], step= self.step)
                    #wandb.log(self.logs[2], step= self.step)

           

    def validation(self, modality):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Step {self.step}:")

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}

        test_data = self.test_dataloader_cif if modality else self.test_dataloader
        iterative = iter(test_data)
        
        

        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            log_dict[f"val_psnr_{inner_steps}_steps_{modality}"] = 0.0
            log_dict[f"val_loss_{inner_steps}_steps_{modality}"] = 0.0

            # Fit modulations for each validation datapoint
            for i in range(len(test_data)-2):
                data = next(iterative).to(self.args.device)
                if self.patcher and not modality:
                    # If using patching, test data will have a batch size of 1.
                    # Remove batch dimension and instead convert data into
                    # patches, with patch dimension acting as batch size
                    patches, spatial_shape = self.patcher.patch(data[0])
                    coordinates, features = self.converter[modality].to_coordinates_and_features(
                        patches
                    )

                    # As num_patches may be much larger than args.batch_size,
                    # split the fitting of patches into batch_size chunks to
                    # reduce memory
                    outputs = metalearning.outer_step_chunked(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        chunk_size=self.args.batch_size,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                        modality=modality
                    )

                    # Shape (num_patches, *patch_shape, feature_dim)
                    patch_features = outputs["reconstructions"]

                    # When using patches, we cannot directly use psnr and loss
                    # output by outer step, since these are calculated on the
                    # padded patches. Therefore we need to reconstruct the data
                    # in its original unpadded form and manually calculate mse
                    # and psnr
                    # Shape (num_patches, *patch_shape, feature_dim) ->
                    # (num_patches, feature_dim, *patch_shape)
                    patch_data = conversion.features2data(patch_features, batched=True)
                    # Shape (feature_dim, *spatial_shape)
                    data_recon = self.patcher.unpatch(patch_data, spatial_shape)
                    # Calculate MSE and PSNR values and log them
                    mse = losses.mse_fn(data_recon, data[0])
                    psnr = losses.mse2psnr(mse)
                    log_dict[f"val_psnr_{inner_steps}_steps_{modality}"] += psnr.item()
                    log_dict[f"val_loss_{inner_steps}_steps_{modality}"] += mse.item()
                else:
                    coordinates, features = self.converter[modality].to_coordinates_and_features(
                        data
                    )

                    outputs = metalearning.outer_step(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        is_train=False,
                        return_reconstructions=True,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                        modality=modality
                    )

                    log_dict[f"val_psnr_{inner_steps}_steps_{modality}"] += outputs["psnr"]
                    log_dict[f"val_loss_{inner_steps}_steps_{modality}"] += outputs["loss"].item()

                if not full_validation and i >= num_validation_batches - 1:
                    break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"val_psnr_{inner_steps}_steps_{modality}"] /= i + 1
            log_dict[f"val_loss_{inner_steps}_steps_{modality}"] /= i + 1

            mean_psnr, mean_loss = (
                log_dict[f"val_psnr_{inner_steps}_steps_{modality}"],
                log_dict[f"val_loss_{inner_steps}_steps_{modality}"],
            )
            print(
                f"Inner steps {inner_steps}, Loss {mean_loss:.3f}, PSNR {mean_psnr:.3f}"
            )

            # Use first setting of inner steps for best validation PSNR
            if inner_steps == self.args.validation_inner_steps[0]:
                if mean_psnr > self.best_val_psnr:
                    self.best_val_psnr = mean_psnr
                    # Optionally save new best model
                    if self.args.use_wandb and self.model_path:
                        torch.save(
                            {
                                "args": self.args,
                                "model_state_dict": self.func_rep.state_dict(),
                                'optimizer_state_dict': self.outer_optimizer.state_dict(),
                                'step': self.step,
                                'epoch': self.epoch,
                            },
                            self.model_path,
                        )

            if self.args.use_wandb:
                # Store final batch of reconstructions to visually inspect model
                # Shape (batch_size, channels, *spatial_dims)
                reconstruction = self.converter[modality].to_data(
                    None, outputs["reconstructions"]
                )
                if self.patcher and not modality:
                    # If using patches, unpatch the reconstruction
                    # Shape (channels, *spatial_dims)
                    reconstruction = self.patcher.unpatch(reconstruction, spatial_shape)

                if modality==0:
                    # Currently only support audio saving when using patches
                    if self.patcher:
                        # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                        if self.test_dataloader.dataset.normalize:
                            reconstruction = 2 * reconstruction - 1
                        # Saved audio sample needs shape (num_samples, num_channels),
                        # so transpose
                        log_dict[
                            f"val_reconstruction_{inner_steps}_steps_{modality}"
                        ] = wandb.Audio(
                            reconstruction.T.cpu(),
                            sample_rate=self.test_dataloader.dataset.sample_rate,
                        )
                else:
                    #print("entered Image reconstruction!!!")
                    log_dict[f"val_reconstruction_{inner_steps}_steps_{modality}"] = wandb.Image(
                        reconstruction
                    )

                wandb.log(log_dict, step=self.step)

        print("\n")

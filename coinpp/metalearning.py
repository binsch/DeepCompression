import coinpp.losses as losses
import coinpp.conversion as conversion
import torch
import torch.utils.checkpoint as cp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def inner_loop(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    is_test=True,
    gradient_checkpointing=False,
    do_sampling=False,
    do_bootstrapping=False,
    inner_steps_boot=3,
    data_ratio=0.5
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    if do_sampling == True:
        sampled_coordinates, sampled_index= gradncp_sample(features, func_rep, data_ratio)
        features_tmp = rearrange(features, 'b h w c -> b c (h w)')
        inputs = torch.gather(features_tmp, 2, sampled_index)
        sampled_features = rearrange(inputs, 'b c s -> b s c')
    
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
            )
        else:
            fitted_modulations = inner_loop_step(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                sampled_coordinates,
                sampled_features,
                inner_lr,
                is_train,
                is_test,
                gradient_checkpointing,
            )

    if do_bootstrapping:
      fitted_mods_L = fitted_modulations
      # perform on full context set
      for step in range(inner_steps_boot):
              fitted_modulations = inner_loop_step(
                  func_rep,
                  fitted_modulations,
                  coordinates,
                  features,
                  coordinates,
                  features,
                  inner_lr,
                  is_train=False,
                  is_test=False,
                  gradient_checkpointing=False,
              )
      fitted_mods_K = fitted_modulations
      return fitted_mods_L, fitted_mods_K
    
    return fitted_modulations, None


def inner_loop_step(
    func_rep,
    modulations,
    coordinates,
    features,
    sampled_coordinates,
    sampled_features,
    inner_lr,
    is_train=False,
    is_test=True,
    gradient_checkpointing=False,
    data_ratio=0.5
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = len(features)

    

    with torch.enable_grad():
        features_recon = func_rep.modulated_forward(sampled_coordinates, modulations)
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        loss = losses.mse_fn(features_recon, sampled_features) * batch_size
        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
    
    #gradient rescaling (only at test time)
    grads_scale = 1
    if is_train==False and is_test==True:
      subsample_grad = grad

      with torch.enable_grad():
          features_recon = func_rep.modulated_forward(coordinates, modulations)
          loss = losses.mse_fn(features_recon, features)* batch_size

          grad = torch.autograd.grad(
              loss,
              modulations,
              create_graph=is_train and not detach,
              allow_unused=True
          )[0]
          subsample_grad_norm= torch.norm(
                  subsample_grad.data.view(batch_size, -1), p=2, dim=1, keepdim=True
              )
          grads_norm = torch.norm(
                  grad.data.view(batch_size, -1), p=2, dim=1, keepdim=True
              )
          grads_scale = subsample_grad_norm / (grads_norm + 1e-16)

    
    # Perform single gradient descent step
    return modulations - func_rep.inner_lr * grads_scale * grad


def outer_step(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    is_test=True,
    return_reconstructions=False,
    gradient_checkpointing=False,
    do_sampling=False,
    do_bootstrapping=False,
    inner_steps_boot=3,
    data_ratio=0.5,
    loss_boot_weight=1.
):
    """
    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """
    func_rep.zero_grad()
    batch_size = len(coordinates)
    modulations_init = torch.zeros(
        batch_size, func_rep.modulation_net.latent_dim, device=coordinates.device
    ).requires_grad_()

    # Run inner loop
    modulations, modulations_boot = inner_loop(
        func_rep,
        modulations_init,
        coordinates,
        features,
        inner_steps,
        inner_lr,
        is_train,
        is_test,
        gradient_checkpointing,
        do_sampling,
        do_bootstrapping,
        inner_steps_boot,
        data_ratio
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        # While the loss in the inner loop is individual for each set of
        # modulations, the loss in the outer loop does depend on the entire
        # batch (we update the base network such that all modulations can easily
        # be fit). We therefore take the mean across the batch dimension so the
        # loss is invariant to the number of elements in the batch
        # Shape (batch_size,)
        per_example_loss = losses.batch_mse_fn(features_recon, features)
        # Shape (1,)
        loss = per_example_loss.mean()

    if do_bootstrapping:
      loss_boot = loss_boot_weight*param_consistency(modulations, modulations_boot, batch_size)
      loss = loss + loss_boot

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = features_recon

    return outputs


def outer_step_chunked(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    chunk_size,
    gradient_checkpointing=False,
):
    """Performs outer step in chunks to reduce memory requirements when a
    datapoint has a large number of patches.

    Args:
        coordinates (torch.Tensor): Shape (num_patches, *, coordinate_dim).
        features (torch.Tensor): Shape (num_patches, *, feature_dim).
        chunk_size (int): Size of chunks to use when fitting inner loop.
            Typically chunk_size < num_patches in order to reduce memory
            requirements.

    Notes:
        This should only be used for validation, not training. Note also that
        this function should only be used for patching, when a large number
        of patches represents a single datapoint. In other cases, batch size
        can just directly be reduced. This function only returns reconstructions
        and modulations.
    """
    # Calculate number of batches of size chunk_size needed to process datapoint
    num_patches = coordinates.shape[0]
    num_batches = num_patches // chunk_size
    last_batch_size = num_patches % chunk_size

    # Fit patches separately and stack them
    reconstructions = []
    modulations = []
    idx = 0
    for _ in range(num_batches):
        next_idx = idx + chunk_size
        outputs = outer_step(
            func_rep,
            coordinates[idx:next_idx],
            features[idx:next_idx],
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        # Shape (chunk_size, *, feature_dim)
        reconstructions.append(outputs["reconstructions"])
        # Shape (chunk_size, latent_dim)
        modulations.append(outputs["modulations"])
        idx = next_idx

    # If non zero final batch size, fit final piece of data
    if last_batch_size:
        outputs = outer_step(
            func_rep,
            coordinates[idx:],
            features[idx:],
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        reconstructions.append(outputs["reconstructions"])
        modulations.append(outputs["modulations"])

    # Reconstructions shape (num_patches, *, feature_dim)
    # Modulations shape (num_patches, latent_dim)
    return {
        "reconstructions": torch.cat(reconstructions, dim=0),
        "modulations": torch.cat(modulations, dim=0),
    }


#Functions to perform and aid in sampling for GradNCP

def shape_to_coords(spatial_shape):
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(-1.0, 1.0, spatial_shape[i]))
    return torch.stack(torch.meshgrid(*coords), dim=-1)

def gradncp_sample(inputs, func_rep, data_ratio=0.5):
    ratio = data_ratio
    coords = rearrange(conversion.shape2coordinates((inputs.shape[1], inputs.shape[2])), 'h w c -> (h w) c')
    meta_batch_size = inputs.size(0)
    coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape)).to("cuda")
    with torch.no_grad():
        out, feature = func_rep(coords, get_penult_features=True)
        if 'img' in ['img']:
            out = rearrange(out, 'b hw c -> b c hw')
            feature = rearrange(feature, 'b hw f -> b f hw')
            inputs = rearrange(inputs, 'b h w c -> b c (h w)')
        else:
            raise NotImplementedError()
        error = inputs - out  # b c (hw)
        gradient = -1 * feature.unsqueeze(dim=1) * error.unsqueeze(dim=2)  # b c f hw
        gradient_bias = -1 * error.unsqueeze(dim=2)  # b c hw
        gradient = torch.cat([gradient, gradient_bias], dim=2)
        gradient = rearrange(gradient, 'b c f hw -> b (c f) hw')
        gradient_norm = torch.norm(gradient, dim=1)  # b hw
        coords_len = gradient_norm.size(1)

    gradncp_index = torch.sort(
        gradient_norm, dim=1, descending=True
    )[1][:, :int(coords_len * ratio)]  # b int(hw * ratio)

    gradncp_coord = torch.gather(
        coords, 1, gradncp_index.unsqueeze(dim=2).repeat(1, 1, 2)
    )
    gradncp_index = gradncp_index.unsqueeze(dim=1).repeat(1, 3, 1)

    return gradncp_coord, gradncp_index

def random_sample(inputs):
    grid = rearrange(conversion.shape2coordinates((inputs.shape[1], inputs.shape[2])), 'h w c -> (h w) c')
    coord_size = grid.size(0)  # shape (h * w, c)
    perm = torch.randperm(coord_size)

def param_consistency(params, params_bootstrap, bs):
    updated_param = params_bootstrap.detach() - params
    updated_param = updated_param.view(bs, -1)
    param_norm = torch.norm(updated_param, p=2, dim=1).mean()
    return param_norm


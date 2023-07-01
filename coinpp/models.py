# Based on https://github.com/EmilienDupont/coin
import torch
from torch import nn
from math import sqrt


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model. If it is, no
            activation is applied and 0.5 is added to the output. Since we
            assume all training data lies in [0, 1], this allows for centering
            the output of the model.
        use_bias (bool): Whether to learn bias in linear layer.
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_last=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0.5
        else:
            out = self.activation(out)
        return out
    
    def subnetwork_forward(self, x, G_low):
        # same as forward, but pointwise multiply the weights with G_low
        # (batch_size, num_points, dim_hidden) x (batch_size, dim_hidden, dim_hidden) -> (batch_size, num_points, dim_hidden)
        if self.linear.bias is not None:
            out = torch.einsum('bni,bhi->bnh', x, ((G_low + 1.0)*self.linear.weight)) + self.linear.bias
        else:
            out = torch.einsum('bni,bhi->bnh', x, ((G_low + 1.0)*self.linear.weight))
        if self.is_last:
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            out += 0.5
        else:
            out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        self.net = nn.Sequential(*layers)

        self.last_layer = SirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias, is_last=True
        )

    def forward(self, x):
        """Forward pass of SIREN model.

        Args:
            x (torch.Tensor): Tensor of shape (*, dim_in), where * means any
                number of dimensions.

        Returns:
            Tensor of shape (*, dim_out).
        """
        x = self.net(x)
        return self.last_layer(x)


class ModulatedSiren(Siren):
    """Modulated SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool): Whether to learn bias in linear layer.
        latent_dim (int): Dimension of latent vector.
        modulation_net_dim_hidden (int): Hidden dimension of modulation net.
        modulation_net_num_res_blocks (int): Number of ResBlocks of the modulation net.
        UV_rank (int): Width of the matrices U and V (d in the VC-INR paper).
        modulate_last_layer (bool): Whether to modulate the last layer.
        use_batch_norm (bool): Whether to use batch norm in ResBlock.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        latent_dim=64,
        modulation_net_dim_hidden=64,
        modulation_net_num_res_blocks=2,
        UV_rank=5,
        modulate_last_layer=True,
        use_batch_norm=True,
    ):
        super().__init__(
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0,
            w0_initial,
            use_bias,
        )
        self.w0 = w0
        self.w0_initial = w0_initial

        self.modulate_last_layer = modulate_last_layer
        self.num_hidden_modulation_matrices = num_layers - 2

        self.modulation_net = LatentToModulationMatrices(
            latent_dim=latent_dim,
            num_hidden_modulation_matrices=self.num_hidden_modulation_matrices,
            modulation_net_dim_hidden=modulation_net_dim_hidden,
            modulation_net_num_res_blocks=modulation_net_num_res_blocks,
            UV_rank=UV_rank,
            siren_dim_hidden=dim_hidden,
            use_batch_norm=use_batch_norm,
            siren_dim_out=(self.dim_out if modulate_last_layer else None),
        )

    def define_inner_lr_params(self, latent_dim, device):
      self.inner_lr = nn.Parameter( torch.ones(latent_dim, requires_grad=True).to(device))
      print(self.inner_lr.is_leaf)


    def modulated_forward(self, x, latent):
        """Forward pass of modulated SIREN model.

        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.

        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        # Extract batch_size and spatial dims of x, so we can reshape output
        x_shape = x.shape[:-1]
        # Flatten all spatial dimensions, i.e. shape
        # (batch_size, *, dim_in) -> (batch_size, num_points, dim_in)
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = self.net[0](x)
        # U, V: (batch_size, num_hidden_layers, dim_hidden, dim_hidden)
        # last_layer_mod: (batch_size, num_hidden_layers, dim_out, dim_hidden)
        U, V, last_layer_mod = self.modulation_net(latent)

        # Iterate through layers and apply corresponding modulation matrix to each.
        for i, module in enumerate(self.net[1:]):
            G_low = nn.functional.sigmoid(torch.einsum('bij,bkj->bik', U[:,i], V[:,i]))
            x = module.subnetwork_forward(x, G_low) # (batch_size, num_points, dim_hidden)

        # Shape (batch_size, num_points, dim_out)
        if self.modulate_last_layer:
            out = self.last_layer.subnetwork_forward(x, last_layer_mod)
        else:
            out = self.last_layer(x)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return out.view(*x_shape, out.shape[-1])


class LatentToModulationMatrices(nn.Module):
    """Maps a latent vector to a set of modulation matrices.

    Args:
        latent_dim (int):
        num_hidden_modulation_matrices (int):
        modulation_net_dim_hidden (int):
        modulation_net_num_res_blocks (int):
        UV_rank (int): Width of U, V (d in the VC-INR paper).
        siren_dim_hidden (list): Hidden dimension m of the ModulatedSiren hidden weights (mxm).
        use_batch_norm (bool): Whether to use batch norm.
        siren_dim_out (int): Dimension of the output of the ModulatedSiren net.
    """

    def __init__(self, latent_dim, num_hidden_modulation_matrices, modulation_net_dim_hidden, modulation_net_num_res_blocks,
                 UV_rank, siren_dim_hidden, use_batch_norm=True, siren_dim_out=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_hidden_modulation_matrices = num_hidden_modulation_matrices
        self.modulation_net_dim_hidden = modulation_net_dim_hidden
        self.UV_rank = UV_rank
        self.siren_dim_hidden = siren_dim_hidden
        self.siren_dim_out = siren_dim_out

        # U, V are 2 matrices of dim (siren_dim_hidden x UV_rank) for each hidden layer of the ModulatedSiren
        self.UV_end_idx = 2 * siren_dim_hidden * UV_rank * num_hidden_modulation_matrices
        # If the last layer is also modulated, we need to generate an additional modulation matrix for it
        # This matrix can be calculated directly (not matrix multiplication necessary),
        # because it is low dimensional (siren_dim_hidden x siren_dim_out) where siren_dim_out is usually in the single digits.
        self.modulation_net_dim_out = (
            self.UV_end_idx +
            (self.siren_dim_out * self.siren_dim_hidden if self.siren_dim_out is not None else 0)
        )

        # The initial layer projects the latent space representation onto a vector that can be used as input for the first ResBlock
        layers = [nn.Linear(latent_dim, modulation_net_dim_hidden), nn.LeakyReLU()]

        # Generate ResBlocks
        for _ in range(modulation_net_num_res_blocks):
            layers.append(ResBlock(modulation_net_dim_hidden, use_batch_norm=use_batch_norm))

        # The last layer maps from the hidden dim of the ResBlock to the correct output length
        layers += [nn.Linear(modulation_net_dim_hidden, self.modulation_net_dim_out)]
        self.net = nn.Sequential(*layers)

        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, latent):
        out = self.layer_norm(latent)
        out = self.net(out)

        UV = out[:,:self.UV_end_idx].view(
            latent.shape[0],
            self.num_hidden_modulation_matrices,
            2,
            self.siren_dim_hidden,
            self.UV_rank
        )
        U, V = UV[:,:,0,:,:], UV[:,:,1,:,:]

        if self.siren_dim_out is None:
            return U, V, None

        last_layer_mod = out[:,self.UV_end_idx:].view(
            latent.shape[0],
            self.siren_dim_out,
            self.siren_dim_hidden
        )

        last_layer_mod = nn.functional.sigmoid(last_layer_mod)

        return U, V, last_layer_mod
        


class ResBlock(nn.Module):
    def __init__(self, dim_hidden, activation=nn.LeakyReLU, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm=use_batch_norm

        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.activation1 = activation()
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        if use_batch_norm:
            self.batchnorm1 = torch.nn.BatchNorm1d(dim_hidden)
            self.batchnorm2 = torch.nn.BatchNorm1d(dim_hidden)
        self.activation2 = activation()

    def forward(self, x):
        residual = x
        if self.use_batch_norm:
            x = self.activation1(self.batchnorm1(self.linear1(x)))
            x = self.batchnorm2(self.linear2(x))
        else:
            x = self.activation1(self.linear1(x))
            x = self.linear2(x)
        output = residual + x
        output = self.activation2(output)
        return output


if __name__ == "__main__":
    dim_in, dim_hidden, dim_out, num_layers = 2, 5, 3, 4
    batch_size, latent_dim = 3, 7
    modulation_matrix_width = 3
    model = ModulatedSiren(
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        modulate_scale=False,
        use_latent=True,
        latent_dim=latent_dim,
        modulation_matrix_width=modulation_matrix_width
    )
    print(model)
    latent = torch.rand(batch_size, latent_dim)
    x = torch.rand(batch_size, 5, 5, 2)
    out = model(x)
    out = model.modulated_forward(x, latent)
    print(out.shape)

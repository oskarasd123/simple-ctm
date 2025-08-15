import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchvision


def norm(x : Tensor):
    return F.rms_norm(x, (x.size(-1),))

def conv_norm(x : Tensor):
    return x * (1/(x.pow(2).mean(1).sqrt()[:, None, :, :] + 1e-5))

class ConvNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))
    
    def forward(self, x):
        return conv_norm(x) * self.gamma[None, :, None, None] + self.beta[None, :, None, None]


class NeuronLevelModels(nn.Module):
    def __init__(self, neurons, history_size, mid_size): # for each neuron there is history_size -> mid_size, mid_size -> 1 
        super().__init__()
        self.neurons = neurons
        self.history_size = history_size
        self.mid_size = mid_size
        self.layer1_weights = nn.Parameter(torch.randn(neurons, history_size, mid_size) / history_size ** 0.5)
        self.layer1_weights.data *= torch.arange(history_size)[None, :, None] / history_size + 1e-3
        self.layer2_weights = nn.Parameter(torch.randn(neurons, mid_size, 1) / mid_size ** 0.5)
        self.layer2_weights.data *= torch.arange(mid_size)[None, :, None] / mid_size + 1e-3
    
    def forward(self, x : Tensor):
        # x has shape (batch, neuron, history)
        x = torch.einsum("bnh, nhm -> bnm", x, self.layer1_weights)
        x = F.gelu(x)
        nn.GELU()
        x = torch.einsum("bnm, nms -> bns", x, self.layer2_weights).squeeze(-1)
        return x



# implemented from paper https://arxiv.org/pdf/2505.05522 Continuous Thought Machines
class CTM(nn.Module):
    def __init__(self, neurons : int, depth : int, input_size : int, mlp_ratio : float):
        super().__init__()
        self.neurons = neurons
        self.depth = depth # lenght of history
        self.input_size = input_size
        # fwd residual block
        mid_width = int(neurons * mlp_ratio)
        self.fwd_residual_linear = nn.Linear(neurons + input_size, neurons)
        self.fwd_compress_linear = nn.Linear(neurons + input_size, mid_width)
        self.fwd_expand_linear = nn.Linear(mid_width, neurons)
        self.fwd_expand_linear.weight.data *= 0.2

        self.neuron_level_models = NeuronLevelModels(neurons, depth, 10)
        self.base_pre_activations = nn.Parameter(torch.randn((1, neurons, depth)))
        self.base_activations = nn.Parameter(torch.randn((1, neurons, depth)))
        self.time_discounts = nn.Parameter(torch.ones((1, neurons, 1)) * 0.9)

        self.pre_activations : Tensor | None = None
        self.activations : Tensor | None = None
        self.latent : Tensor | None = None # is also output
        self.__call__ = self.forward
    
    def reset(self, batch_size : int):
        self.pre_activations = self.base_pre_activations.repeat((batch_size, 1, 1)) # repeat batch dimension
        self.activations = self.base_activations.repeat((batch_size, 1, 1))
        self.compute_latent()
        return self.latent
    
    def forward(self, x : Tensor):
        x = torch.cat([self.activations[:, :, -1], x[:, :]], 1)
        mid = self.fwd_compress_linear(x)
        x = self.fwd_expand_linear(F.gelu(mid)) + self.fwd_residual_linear(x)
        x = norm(x)
        self.pre_activations = torch.cat([self.pre_activations, x[:, :, None]], 2)[:, :, 1:]
        new_activations = self.neuron_level_models(self.pre_activations)
        self.activations = torch.cat([self.activations, new_activations[:, :, None]], -1)[:, :, 1:]
        return self.compute_latent()
        
    def compute_latent(self):
        prod = self.activations * self.activations.roll(1, 1)
        discount = ((self.time_discounts) ** torch.arange(0, self.depth, device=self.activations.device)[None, None, :]).flip(-1)
        discount /= discount.sum(-1, keepdim=True)
        self.latent = (prod * discount).sum(-1)
        return self.latent
    

class SimpleCTM(nn.Module):
    def __init__(self, neurons : int, depth : None, input_size : int, mlp_ratio : float):
        super().__init__()
        self.neurons = neurons
        self.input_size = input_size

        mid_width = int(neurons * mlp_ratio)
        self.fwd_residual_linear = nn.Linear(neurons + input_size, neurons)
        self.fwd_compress_linear = nn.Linear(neurons + input_size, mid_width)
        self.fwd_expand_linear = nn.Linear(mid_width, neurons)
        self.base_latent = nn.Parameter(torch.randn((neurons,)))
        self.latent = None
    
    def reset(self, batch_size):
        self.latent = self.base_latent.unsqueeze(0).repeat((batch_size, 1))
        return self.latent

    def forward(self, x : Tensor):
        x = torch.cat([self.latent, x], 1)
        mid = self.fwd_compress_linear(x)
        x = self.fwd_expand_linear(F.tanh(mid)) + self.fwd_residual_linear(x)
        self.latent = norm(x)
        return self.latent





class ImagePositionalEmbedding(nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        self.multipliers = nn.Parameter(torch.ones((dim,)))
    
    def forward(self, image : Tensor):
        x = torch.arange(0, image.size(2), 1)/image.size(2) * torch.pi
        y = torch.arange(0, image.size(3), 1)/image.size(3) * torch.pi

        add_field = torch.zeros((image.shape[1:]), device = image.device)
        dim = image.size(1)
        add_field[0::4] = (torch.arange(dim//4)[:, None] * x[None, :]).cos()[:, :, None]
        add_field[1::4] = (torch.arange(dim//4)[:, None] * x[None, :]).sin()[:, :, None]
        add_field[2::4] = (torch.arange(dim//4)[:, None] * y[None, :]).cos()[:, None, :]
        add_field[3::4] = (torch.arange(dim//4)[:, None] * y[None, :]).sin()[:, None, :]

        return image + add_field.unsqueeze(0) * self.multipliers[None, :, None, None]


        
        


class CTMImageAttention(nn.Module):
    def __init__(self, channels : int, latent_dim : int, output_dim : int):
        super().__init__()
        self.q_linear = nn.Linear(latent_dim, output_dim)
        self.k_linear = nn.Linear(channels, output_dim)
        self.v_linear = nn.Linear(channels, output_dim)
        self.embedding = ImagePositionalEmbedding(channels)
    
    def forward(self, image_features : Tensor, latent : Tensor, return_attn = False):
        q = self.q_linear(latent)
        image_features = self.embedding(image_features)
        image_features = image_features.permute((0, 2, 3, 1)) # now is (b, w, h, channels)
        k = self.k_linear(image_features)
        v = self.v_linear(image_features)
        if not return_attn:
            o = F.scaled_dot_product_attention(q.unsqueeze(1), norm(k).flatten(1, 2), norm(v).flatten(1, 2)).squeeze(1)
            return o
        attn_weights = torch.einsum("bn, bxyn -> bxy", q, norm(k)) / q.size(-1)**0.5
        attn_weights = F.softmax(attn_weights.flatten(1, 2), 1).reshape_as(attn_weights)
        o = torch.einsum("bxy, bxyc -> bc", attn_weights, norm(v))
        return o, attn_weights
    
    def __call__(self, image_features : Tensor, latent : Tensor, return_attn = False):
        return self.forward(image_features, latent, return_attn)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.mid_channels = int((in_channels + out_channels) * mid_ratio / 2)
        self.up_conv = nn.Conv2d(in_channels, self.mid_channels, 3, 1, 1)
        self.down_conv = nn.Conv2d(self.mid_channels, out_channels, 1, 1, 0)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.output_norm = nn.GroupNorm(out_channels, out_channels)
    
    def forward(self, x):
        out = self.residual_conv(x) + self.down_conv(F.gelu(self.up_conv((x))))
        return self.output_norm(out)


class ConvBlockDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, mid_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.mid_channels = int((in_channels + out_channels) * mid_ratio / 2)
        self.up_conv = nn.Conv2d(in_channels, self.mid_channels, 3, 1, 1)
        self.down_conv = nn.Conv2d(self.mid_channels, out_channels, 4, 2, 1) # down samples by 2x
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 2, 2, 0) # also down samples by 2x, but uses 2x2 kernel
        self.output_norm = nn.GroupNorm(out_channels, out_channels)
    
    def forward(self, x):
        assert x.size(2) % 2 == 0 and x.size(3) % 2 == 0, f"got input shape: {x.shape}. This module doesn't handle padding for uneven inputs"
        out = self.residual_conv(x) + self.down_conv(F.gelu(self.up_conv((x))))
        return self.output_norm(out)

class ImageExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 256
        self.layers = nn.Sequential(*[
            ConvBlockDownSample(3, 32, 2), # 2x decrease in resolution
            ConvBlock(32, 64, 2),
            ConvBlockDownSample(64, 64, 2), # 4x
            ConvBlockDownSample(64, 128, 2), # 8x 
            ConvBlock(128, 128, 2),
            ConvBlockDownSample(128, 128, 2), # 16x
            ConvBlock(128, 256, 2),
        ])

    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)
    
    def __call__(self, x : Tensor):
        return self.forward(x)


class MoblileConvBlock(nn.Module): # conv block from mobile net
    def __init__(self, in_channels, out_channels, expand_ratio = 2):
        super().__init__()
        mid_channels = int((in_channels + out_channels)/2*expand_ratio)
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, 1, 1)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1, mid_channels) # per channel conv
        self.compress_conv = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, 1, 1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.output_norm = nn.GroupNorm(out_channels, out_channels)
    
    def forward(self, x):
        out = self.residual_conv(x) + self.compress_conv(F.relu(self.mid_conv(F.relu(self.expand_conv(x)))))
        return self.output_norm(out)

class MoblileConvBlockDownSample(nn.Module): # conv block from mobile net
    def __init__(self, in_channels, out_channels, expand_ratio = 2):
        super().__init__()
        mid_channels = int((in_channels + out_channels)/2*expand_ratio)
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, 1, 1)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels, 4, 2, 1, 1, mid_channels) # per channel conv
        self.compress_conv = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, 1, 1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 2, 2)
        self.output_norm = nn.GroupNorm(out_channels, out_channels)
    
    def forward(self, x):
        out = self.residual_conv(x) + self.compress_conv(F.relu(self.mid_conv(F.relu(self.expand_conv(x)))))
        return self.output_norm(out)

class MoblileImageExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 256
        self.layers = nn.Sequential(*[
            MoblileConvBlockDownSample(3, 32, 2), # 2x decrease in resolution
            MoblileConvBlock(32, 64, 2),
            MoblileConvBlockDownSample(64, 64, 2), # 4x
            MoblileConvBlockDownSample(64, 128, 2), # 8x 
            MoblileConvBlock(128, 128, 2),
            MoblileConvBlockDownSample(128, 128, 2), # 16x
            MoblileConvBlock(128, 256, 2),
        ])

    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)
    
    def __call__(self, x : Tensor):
        return self.forward(x)



if __name__ == "__main__":
    image_attention = CTMImageAttention(64, 64, 64)
    latent = torch.randn((1, 64))
    image_features = torch.randn((1, 64, 5, 5))

    o1 = image_attention(image_features, latent)
    o2, attn_weights = image_attention(image_features, latent, True)
    assert (o1-o2).std() < 1e-4 and (o1-o2).abs().max() < 1e-3, f"image attention test failed: std: {(o1-o2).std()}, max: {(o1-o2).abs().max()}"
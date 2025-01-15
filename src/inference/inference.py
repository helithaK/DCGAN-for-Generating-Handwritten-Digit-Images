import torch

def generate_images(generator, latent_dim, num_samples=25, device="cpu"):
    z = torch.randn(num_samples, latent_dim).to(device)
    gen_imgs = generator(z)
    return gen_imgs

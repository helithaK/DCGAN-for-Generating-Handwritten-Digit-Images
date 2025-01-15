import os
import torchvision.utils as vutils

def save_images(images, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    grid = vutils.make_grid(images, normalize=True)
    vutils.save_image(grid, f"{output_dir}/epoch_{epoch}.png")

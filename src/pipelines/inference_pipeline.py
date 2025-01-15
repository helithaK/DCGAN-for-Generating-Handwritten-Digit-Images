import os
import torch
from src.config.config import Config
from src.models.generator import Generator
from src.inference.inference import generate_images
from src.utils.visualizer import save_images

def run_inference_pipeline():
    # Load configuration
    config = Config()
    
    # Set up model and load checkpoint
    img_shape = (config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    generator = Generator(config.LATENT_DIM, img_shape)
    generator_path = os.path.join(config.MODEL_DIR, "generator.pth")

    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator model not found at {generator_path}")

    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    # Generate images
    gen_imgs = generate_images(generator, config.LATENT_DIM, num_samples=25, device="cuda" if torch.cuda.is_available() else "cpu")

    # Save generated images
    save_images(gen_imgs, epoch="inference", output_dir=config.OUTPUT_DIR)

if __name__ == "__main__":
    run_inference_pipeline()

from src.config.config import Config
from src.data.data_loader import get_dataloader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import train_gan

if __name__ == "__main__":
    config = Config()
    dataloader = get_dataloader(config.DATA_DIR, config.BATCH_SIZE)

    img_shape = (config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    generator = Generator(config.LATENT_DIM, img_shape)
    discriminator = Discriminator(img_shape)

    train_gan(config, generator, discriminator, dataloader)

from src.config.config import Config
from src.data.data_loader import get_dataloader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import train_gan
from src.utils.logger import setup_logger

def run_training_pipeline():
    # Set up logger
    logger = setup_logger(log_dir="outputs/logs")
    logger.info("Training pipeline started.")

    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully.")

        # Load dataset
        dataloader = get_dataloader(config.DATA_DIR, config.BATCH_SIZE)
        logger.info("Data loader initialized.")

        # Initialize models
        img_shape = (config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
        generator = Generator(config.LATENT_DIM, img_shape)
        discriminator = Discriminator(img_shape)
        logger.info("Models initialized.")

        # Train GAN
        train_gan(config, generator, discriminator, dataloader)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training pipeline: {e}", exc_info=True)

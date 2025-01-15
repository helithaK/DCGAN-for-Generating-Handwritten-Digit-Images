import torch
import torch.optim as optim
from src.utils.visualizer import save_images
from src.training.loss import get_adversarial_loss

def train_gan(config, generator, discriminator, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    adversarial_loss = get_adversarial_loss()

    for epoch in range(config.EPOCHS):
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            real_imgs = imgs.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), config.LATENT_DIM).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
        
        save_images(gen_imgs.data[:25], epoch, config.OUTPUT_DIR)
        print(f"[Epoch {epoch}] Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")

        # Save the generator model checkpoint
        torch.save(generator.state_dict(), f"{config.MODEL_DIR}/generator.pth")

import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

def calculate_inception_score(images, batch_size=32, splits=10):
    """
    Calculates the Inception Score (IS) for generated images.

    Args:
        images (torch.Tensor): Tensor of images, shape (N, C, H, W).
        batch_size (int): Batch size for processing images through the Inception model.
        splits (int): Number of splits for calculating the IS.

    Returns:
        float: Inception Score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Resize images to (299, 299) for Inception model
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear")
    images = images.to(device)

    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with torch.no_grad():
            pred = inception_model(batch)
            preds.append(pred.softmax(dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    split_scores = []

    for k in range(splits):
        part = preds[k * (len(preds) // splits):(k + 1) * (len(preds) // splits), :]
        p_y = np.mean(part, axis=0)
        scores = np.sum(part * np.log(part / p_y[None, :]), axis=1)
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_fid(real_images, fake_images):
    """
    Calculates the Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_images (torch.Tensor): Real images, shape (N, C, H, W).
        fake_images (torch.Tensor): Generated images, shape (M, C, H, W).

    Returns:
        float: Fréchet Inception Distance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    def get_features(images):
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear")
        images = images.to(device)
        with torch.no_grad():
            features = inception_model(images).detach()
        return adaptive_avg_pool2d(features, (1, 1)).squeeze()

    real_features = get_features(real_images).cpu().numpy()
    fake_features = get_features(fake_images).cpu().numpy()

    # Calculate mean and covariance
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Calculate FID score
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def discriminator_accuracy(discriminator, real_images, fake_images):
    """
    Calculates the discriminator's accuracy in distinguishing real and fake images.

    Args:
        discriminator (torch.nn.Module): The trained discriminator model.
        real_images (torch.Tensor): Real images, shape (N, C, H, W).
        fake_images (torch.Tensor): Generated images, shape (M, C, H, W).

    Returns:
        dict: Accuracy on real and fake images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = discriminator.to(device)
    real_images, fake_images = real_images.to(device), fake_images.to(device)

    with torch.no_grad():
        real_preds = discriminator(real_images)
        fake_preds = discriminator(fake_images)

    real_acc = (real_preds > 0.5).float().mean().item()
    fake_acc = (fake_preds < 0.5).float().mean().item()

    return {"real_accuracy": real_acc, "fake_accuracy": fake_acc}

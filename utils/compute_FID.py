import os
import torch
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class FID_score_calculator:
    def __init__(self, batch_size=50, device=None):
        
        
        self.batch_size = batch_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Inception v3 model
        self.inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception.fc = nn.Identity() 
        self.inception.eval()

        # Define transform for images to feed into Inception
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225] 
            )
        ])

        self.real_mu = None
        self.real_sigma = None

    def get_activation(self, image_tensor):
        
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            activation = self.inception(image_tensor)
            activation = activation.cpu().numpy().reshape(-1)
        return activation

    def compute_statistics(self, images):
        
        activations = []
        # Convert everything into inception-sized tensors
        if isinstance(images, list):
            # images are PIL images
            for img in images:
                img_tensor = self.transform(img).unsqueeze(0)
                act = self.get_activation(img_tensor)
                activations.append(act)
        else:
            # images is a torch tensor of shape [N, C, H, W]
            for i in range(images.shape[0]):
                img = transforms.ToPILImage()(images[i])
                img_tensor = self.transform(img).unsqueeze(0)
                act = self.get_activation(img_tensor)
                activations.append(act)

        activations = np.array(activations)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def compute_fid(self, gen_mu, gen_sigma):
        
        if self.real_mu is None or self.real_sigma is None:
            raise ValueError("Real statistics (mu, sigma) are not set. Compute or load them first.")

        # Compute the squared difference between means
        diff = self.real_mu - gen_mu
        diff_squared = np.dot(diff, diff)

        # Compute the sqrt of product of covariance matrices
        covmean, _ = sqrtm(self.real_sigma.dot(gen_sigma), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Handle singular covariance matrices
        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding epsilon to diagonal of covariances")
            eps = 1e-6
            self.real_sigma += np.eye(self.real_sigma.shape[0]) * eps
            gen_sigma += np.eye(gen_sigma.shape[0]) * eps
            covmean, _ = sqrtm(self.real_sigma.dot(gen_sigma), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

        # Compute the trace part
        trace = np.trace(self.real_sigma) + np.trace(gen_sigma) - 2 * np.trace(covmean)

        fid = diff_squared + trace
        return fid

    def set_real_statistics(self, mu, sigma):
        self.real_mu = mu
        self.real_sigma = sigma


def compute_and_save_real_statistics(fid_calculator, data_root='data', num_images=50000, save_dir='stats'):
    
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cifar10_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    data_loader = DataLoader(cifar10_dataset, batch_size=100, shuffle=False)

    activations_list = []
    count = 0
    with torch.no_grad():
        for batch_images, _ in data_loader:
            if count >= num_images:
                break
            needed = min(batch_images.size(0), num_images - count)
            batch_images = batch_images[:needed]
            count += needed

            batch_images = batch_images.to(fid_calculator.device)
            batch_acts = fid_calculator.inception(batch_images).cpu().numpy()
            activations_list.append(batch_acts)

    activations = np.concatenate(activations_list, axis=0)

    real_mu = np.mean(activations, axis=0)
    real_sigma = np.cov(activations, rowvar=False)

    # Save the statistics
    np.save(os.path.join(save_dir, 'real_mu.npy'), real_mu)
    np.save(os.path.join(save_dir, 'real_sigma.npy'), real_sigma)


def disassemble_integrated_image(img_idx, integrated_image_path, 
                                 sub_img_size=32, 
                                 grid_size=10, 
                                 divider_size=0, 
                                 edge_border=0,
                                 output_dir='extracted_images'):
    """
    Disassembles an integrated image into multiple PIL images and saves them individually.
    """
    # Open and convert image
    integrated_image = Image.open(integrated_image_path).convert('RGB')
    width, height = integrated_image.size

    # Calculate expected dimensions
    expected_width = (grid_size * sub_img_size) + (grid_size - 1)*divider_size + 2*edge_border
    expected_height = (grid_size * sub_img_size) + (grid_size - 1)*divider_size + 2*edge_border

    if width != expected_width or height != expected_height:
        raise ValueError(
            f"Integrated image dimensions {width}x{height} do not match "
            f"expected {expected_width}x{expected_height}. "
            "Check your parameters: sub_img_size, grid_size, divider_size, and edge_border."
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    images = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the coordinates of the sub-image
            left = edge_border + j*(sub_img_size + divider_size)
            upper = edge_border + i*(sub_img_size + divider_size)
            right = left + sub_img_size
            lower = upper + sub_img_size
            img = integrated_image.crop((left, upper, right, lower))
            images.append(img)

    # Save each extracted sub-image
    for idx, img in enumerate(images):
        img_path = os.path.join(output_dir, f"subimage_{img_idx}_{idx}.png")
        img.save(img_path)

    return images


if __name__ == '__main__':

    image_dir = 'results/ddim_linear_0.0'

    # Initialize the calculator
    fid_calculator = FID_score_calculator()

    # Path to directory where we save and load stats
    stats_dir = 'stats'
    real_mu_path = os.path.join(stats_dir, 'real_mu.npy')
    real_sigma_path = os.path.join(stats_dir, 'real_sigma.npy')

    # Compute real statistics if not available
    if not (os.path.exists(real_mu_path) and os.path.exists(real_sigma_path)):
        print("Real statistics not found. Computing now...")
        compute_and_save_real_statistics(fid_calculator, data_root='data', num_images=50000, save_dir=stats_dir)

    # Load real statistics
    real_mu = np.load(real_mu_path)
    real_sigma = np.load(real_sigma_path)
    fid_calculator.set_real_statistics(real_mu, real_sigma)

    # Directory containing integrated images

    integrated_image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    total_list = []
    # Compute FID for each integrated image
    for img_idx, img_file in enumerate(integrated_image_files):
        img_path = os.path.join(image_dir, img_file)
        generated_images = disassemble_integrated_image(img_idx, img_path, sub_img_size=32, grid_size=10, divider_size=2,    # if you have dividers, adjust accordingly
        edge_border=2,     # if you have borders, adjust accordingly
        output_dir='my_extracted_images')
        total_list.extend(generated_images)


    # Compute generated statistics
    gen_mu, gen_sigma = fid_calculator.compute_statistics(total_list)

    # Compute FID
    fid = fid_calculator.compute_fid(gen_mu, gen_sigma)
    print(f"FID for {img_file}: {fid}")

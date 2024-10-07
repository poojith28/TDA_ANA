
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import umap
import gudhi as gd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
from torchvision.utils import make_grid
from dadapy import Data

# Set random seed for reproducibility
manualSeed = 999
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)

import umap
import gudhi as gd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def compute_and_save_persistence_real_data(epoch, real_data, image_filenames):
    """
    Generates a point cloud from the real images, computes its persistence diagram,
    and saves visualizations. Also saves a mapping between points and image filenames.
    """
    # Flatten images: [batch_size, channels, height, width] -> [batch_size, features]
    flattened = real_data.view(real_data.size(0), -1).detach().cpu().numpy()

    # Apply UMAP to directly reduce to 2D
    umap_reducer = umap.UMAP(n_components=2)
    point_cloud = umap_reducer.fit_transform(flattened)

    # Save point cloud data with image filenames
    mapping_df = pd.DataFrame({
        'image_filename': image_filenames,
        'umap1': point_cloud[:, 0],
        'umap2': point_cloud[:, 1]
    })
    mapping_csv_path = f'point_clouds/real_point_cloud_epoch_{epoch:03d}.csv'
    mapping_df.to_csv(mapping_csv_path, index=False)

    # Plot and save point cloud
    plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, alpha=0.7, c='green')
    plt.title(f'Real Data Point Cloud at Epoch {epoch}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.tight_layout()
    point_cloud_plot_path = f'point_clouds/real_point_cloud_epoch_{epoch:03d}.png'
    plt.savefig(point_cloud_plot_path)
    plt.close()

    # Compute persistence diagram using Vietoris-Rips complex
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # Save persistence as DataFrame
    persistence_df = persistence_to_dataframe(persistence, image_filenames)

    # Plot persistence diagram
    try:
        gd.plot_persistence_diagram(persistence)
        plt.title(f'Real Data Persistence Diagram at Epoch {epoch}')
        persistence_diagram_plot_path = f'persistence_diagrams/real_persistence_diagram_epoch_{epoch:03d}.png'
        plt.savefig(persistence_diagram_plot_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting persistence diagram for real data at epoch {epoch}: {e}")

    # Save persistence data as CSV
    persistence_csv_path = f'persistence_diagrams/real_persistence_epoch_{epoch:03d}.csv'
    persistence_df.to_csv(persistence_csv_path, index=False)

def persistence_to_dataframe(persistence, image_filenames):
    data = []
    for i, feature in enumerate(persistence):
        dim, (birth, death) = feature
        if death == float('inf'):
            death = None
        data.append({'birth': birth, 'death': death, 'dimension': dim})
    return pd.DataFrame(data)

# Example usage in your training loop
# Assuming `real_data` is a batch of real images from CIFAR-10
# `image_filenames` is a list of filenames for the images in this batch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def calculate_intrinsic_dimension(data):
    """
    Estimate the intrinsic dimension of a dataset using the 2NN method from GUDHI.

    Parameters:
    data (numpy.ndarray): The dataset, where each row is a datapoint.

    Returns:
    float: The estimated intrinsic dimension.
    """
    # Convert the data to a GUDHI Data object and compute the 2NN intrinsic dimension
    id_estimator = Data(data)
    intrinsic_dimension, err, scale = id_estimator.compute_id_2NN()
    
    return intrinsic_dimension

# Hyperparameters
num_epochs = 1000
batch_size = 256
lr_generator = 2e-4  # Increased learning rate for generator
lr_discriminator = 1e-4 # Decreased learning rate for discriminator
latent_size = 128  # Increased latent size
image_size = 32
image_channels = 3

# Create directories to save outputs
os.makedirs("generated_images/individual", exist_ok=True)
os.makedirs("point_clouds", exist_ok=True)
os.makedirs("persistence_diagrams", exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(image_channels)],
                         [0.5 for _ in range(image_channels)])
])

# Load CIFAR-10 dataset
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter out class 0 (airplane)
class_label = 0
class_indices = [i for i, label in enumerate(dataset.targets) if label == class_label]
filtered_dataset = Subset(dataset, class_indices)
print(f'Number of images in class {class_label}: {len(filtered_dataset)}')

# Data loader
dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

epoch = 0  # Replace this with the current epoch number
real_data = next(iter(dataloader))[0]  # Get a batch of real data from your dataloader
image_filenames = [f'real_image_{i:03d}.png' for i in range(real_data.size(0))]  # Generate image filenames

# Call the function to compute and save persistence diagram for real data
compute_and_save_persistence_real_data(epoch, real_data, image_filenames)

# Define weight initialization function
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Model with Increased Parameters
class Generator(nn.Module):
    def __init__(self, latent_size, image_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, 1024, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, image_channels, 3, 1, 1, bias=False),  # 32x32 -> 32x32
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Model with Increased Parameters
class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, 128, 4, 2, 1, bias=False),  # 32x32 ->16x16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16x16 ->8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8x8 ->4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # 4x4 ->2x2
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, 2, 1, 0, bias=False),  # 2x2 ->1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# Initialize models
generator = Generator(latent_size, image_channels).to(device)
discriminator = Discriminator(image_channels).to(device)

# Apply weight initialization
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss and optimizers
criterion = nn.BCELoss()

optimizerD = optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))

# Labels
real_label_smoothing = 0.9  # Label smoothing for real labels
fake_label_noise = 0.1  # Maximum noise for fake labels

# Initialize lists to store losses and intrinsic dimensions
G_losses = []
D_losses = []
real_intrinsic_dims = []
generated_intrinsic_dims = []
def compute_and_save_persistence(epoch, generated_images, image_filenames):
    # Flatten images: [batch_size, channels, height, width] -> [batch_size, features]
    flattened = generated_images.view(generated_images.size(0), -1).detach().cpu().numpy()

    # Apply UMAP to directly reduce to 2D
    umap_reducer = umap.UMAP(n_components=2)
    point_cloud = umap_reducer.fit_transform(flattened)

    # Save point cloud data with image filenames
    mapping_df = pd.DataFrame({
        'image_filename': image_filenames,
        'umap1': point_cloud[:, 0],
        'umap2': point_cloud[:, 1]
    })
    mapping_csv_path = f'point_clouds/point_cloud_epoch_{epoch:03d}.csv'
    mapping_df.to_csv(mapping_csv_path, index=False)

    # Plot and save point cloud
    plt.figure(figsize=(6, 6))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=10, alpha=0.7, c='blue')
    plt.title(f'Point Cloud at Epoch {epoch}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.tight_layout()
    point_cloud_plot_path = f'point_clouds/point_cloud_epoch_{epoch:03d}.png'
    plt.savefig(point_cloud_plot_path)
    plt.close()

    # Compute persistence diagram using Vietoris-Rips complex
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # Save persistence as DataFrame
    persistence_df = persistence_to_dataframe(persistence, image_filenames)

    # Plot persistence diagram
    try:
        gd.plot_persistence_diagram(persistence)
        plt.title(f'Persistence Diagram at Epoch {epoch}')
        persistence_diagram_plot_path = f'persistence_diagrams/persistence_diagram_epoch_{epoch:03d}.png'
        plt.savefig(persistence_diagram_plot_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting persistence diagram at epoch {epoch}: {e}")

    # Save persistence data as CSV
    persistence_csv_path = f'persistence_diagrams/persistence_epoch_{epoch:03d}.csv'
    persistence_df.to_csv(persistence_csv_path, index=False)

def persistence_to_dataframe(persistence, image_filenames):
    data = []
    for i, (dim, (birth, death)) in enumerate(persistence):
        if death == float('inf'):
            death = None
        # Add the image filename associated with this persistence feature
        if i < len(image_filenames):
            filename = image_filenames[i]
        else:
            filename = "Unknown"
        data.append({
            'birth': birth,
            'death': death,
            'dimension': dim,
            'image_filename': filename
        })
    return pd.DataFrame(data)


# Function to generate unique filenames for images
def generate_image_filenames(epoch, batch_idx, num_images):
    return [f'epoch_{epoch:03d}_batch_{batch_idx:04d}_img_{i:03d}.png' for i in range(num_images)]

# Function to plot and save loss graph
def plot_and_save_loss_graph(G_losses, D_losses, filename='loss_graph.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
real_data2 = next(iter(dataloader))[0]
# Training Loop
for epoch in range(1, num_epochs + 1):
    image_dir = 'generated_images/individual'

    # Initialize an empty list to store generated images for this epoch
    all_fake_images = []
    running_loss_G = 0.0
    running_loss_D = 0.0

    for i, (data, _) in enumerate(dataloader, 0):
        b_size = data.size(0)
        
        ############################
        # (1) Update Discriminator
        ###########################
        discriminator.zero_grad()
        real_data = data.to(device)

        # Labels with smoothing and randomness
        label_real = torch.full((b_size,), real_label_smoothing, dtype=torch.float, device=device)
        label_fake = torch.rand(b_size, device=device) * fake_label_noise  # Random fake labels between 0 and noise level

        # Forward pass real data through Discriminator
        output_real = discriminator(real_data)
        errD_real = criterion(output_real, label_real)
        D_x = output_real.mean().item()

        # Generate fake images
        noise = torch.randn(b_size, latent_size, 1, 1, device=device)
        fake_data = generator(noise)

        # Forward pass fake data through Discriminator
        output_fake = discriminator(fake_data.detach())
        errD_fake = criterion(output_fake, label_fake)
        D_G_z1 = output_fake.mean().item()

        # Total Discriminator loss
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update Generator
        ###########################
        generator.zero_grad()
        label_gen = torch.full((b_size,), 1.0, dtype=torch.float, device=device)  # Generator tries to make discriminator believe outputs are real

        # Forward pass fake data through Discriminator
        output_gen = discriminator(fake_data)
        errG = criterion(output_gen, label_gen)
        D_G_z2 = output_gen.mean().item()

        errG.backward()
        optimizerG.step()

        # Store the generated images for later use
        all_fake_images.append(fake_data.detach())

        # Track the losses
        running_loss_D += errD.item()
        running_loss_G += errG.item()

        # Output training stats every 100 steps
        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    # After all batches, calculate average losses for this epoch
    D_losses.append(running_loss_D / len(dataloader))
    G_losses.append(running_loss_G / len(dataloader))

    # Only save data every 25 epochs
    if epoch % 25 == 0:
        # After all batches are processed, concatenate all_fake_images
        all_fake_images = torch.cat(all_fake_images)  # Concatenate all fake images

        # Randomly select 128 images from all_fake_images
        num_images_to_save = min(len(all_fake_images), 128)  # Ensure we don't exceed available images
        selected_indices = random.sample(range(len(all_fake_images)), num_images_to_save)  # Randomly pick 128 indices
        selected_fake_images = all_fake_images[selected_indices]

        # Generate unique filenames for the selected 128 images
        image_filenames = generate_image_filenames(epoch, 0, num_images_to_save)

        # Save the randomly selected 128 generated images for this epoch
        for j in range(num_images_to_save):
            save_image(selected_fake_images[j], os.path.join(image_dir, image_filenames[j]), normalize=True)

        # Compute persistence diagrams for the saved images
        compute_and_save_persistence(epoch, selected_fake_images, image_filenames)
        
        # Calculate intrinsic dimension for real data (flattened)
        real_data_flat = real_data2.view(real_data2.size(0), -1).cpu().numpy()
        real_intrinsic_dim = calculate_intrinsic_dimension(real_data_flat)
        real_intrinsic_dims.append(real_intrinsic_dim)

        # Calculate intrinsic dimension for generated data
        fake_data_flat = all_fake_images.view(all_fake_images.size(0), -1).cpu().numpy()
        generated_intrinsic_dim = calculate_intrinsic_dimension(fake_data_flat)
        generated_intrinsic_dims.append(generated_intrinsic_dim)

        # Save grid of generated images for this epoch
        grid = make_grid(selected_fake_images[:64], nrow=8, normalize=True)
        save_image(grid, f'generated_images/grid_epoch_{epoch:03d}.png', normalize=True)

# Save the final generator and discriminator models after training completes
torch.save(generator.state_dict(), 'dcgan_generator.pth')
torch.save(discriminator.state_dict(), 'dcgan_discriminator.pth')

# Save the losses and intrinsic dimensions to CSV
loss_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'G_loss': G_losses,
    'D_loss': D_losses
})
loss_df.to_csv('losses.csv', index=False)

intrinsic_dim_df = pd.DataFrame({
    'Epoch': range(25, num_epochs + 1, 25),
    'Real_Intrinsic_Dim': real_intrinsic_dims,
    'Generated_Intrinsic_Dim': generated_intrinsic_dims
})
intrinsic_dim_df.to_csv('intrinsic_dimensions.csv', index=False)

# Plot and save loss graph
plot_and_save_loss_graph(G_losses, D_losses)

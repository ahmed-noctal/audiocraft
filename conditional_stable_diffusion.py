import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check device (use MPS for Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
IMG_SIZE = 28
CHANNELS = 1
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-4
T = 1000  # Number of timesteps in the diffusion process

# Noise schedule (linear schedule)
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, device=device)

betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Forward diffusion process
def forward_diffusion(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    if alpha_t.size(0) != x0.size(0):
        alpha_t = alpha_t[:x0.size(0)]
    return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise, noise

# Define U-Net model with conditioning
class UNet(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels + num_classes + 10, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, class_emb):
        # Append class embedding to input
        t_emb = torch.sin(t.float().unsqueeze(1) * torch.arange(0, 10).to(x.device)).unsqueeze(2).unsqueeze(3)
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))  # Ensure t_emb has the same spatial dimensions as x
        class_emb = class_emb.view(class_emb.size(0), -1, 1, 1)
        class_emb = class_emb.expand(-1, -1, x.size(2), x.size(3))  # Ensure class_emb has the same spatial dimensions as x
        x = torch.cat([x, t_emb, class_emb], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Training setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(img_channels=CHANNELS, num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        current_batch_size = images.size(0)
        t = torch.randint(0, T, (current_batch_size,)).to(device)
        x_t, noise = forward_diffusion(images, t)

        # Create class embeddings
        class_emb = torch.zeros((current_batch_size, NUM_CLASSES), device=device)
        class_emb[torch.arange(current_batch_size), labels] = 1.0

        # Predict noise
        predicted_noise = model(x_t, t, class_emb)
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss.item()})

# Sampling loop (generation)
def sample(model, num_samples, labels):
    model.eval()
    x_t = torch.randn((num_samples, CHANNELS, IMG_SIZE, IMG_SIZE), device=device)
    for t in reversed(range(T)):
        t_tensor = torch.tensor([t] * num_samples, device=device)
        class_emb = torch.zeros((num_samples, NUM_CLASSES), device=device)
        class_emb[torch.arange(num_samples), labels] = 1.0
        noise = torch.randn_like(x_t) if t > 0 else 0
        x_t = model(x_t, t_tensor, class_emb) + noise
    return x_t

# Generate conditional samples
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
samples = sample(model, len(labels), labels)

import matplotlib
matplotlib.use('macosx')

import matplotlib.pyplot as plt

# Visualize the generated samples
samples = samples.detach().cpu()
fig, axs = plt.subplots(1, 10, figsize=(15, 3))
for i, ax in enumerate(axs):
    ax.imshow(samples[i, 0], cmap="gray")
    ax.axis("off")
fig.savefig('generated_samples.png')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.datasets import make_moons

# Define hyperparameters
T = 1000  # Total number of timesteps
betas = torch.linspace(0.0001, 0.02, T) # Variance schedule (linear)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # \bar{\alpha}_t in the paper

# Helper function to get the value of \bar{\alpha}_t for a batch of timesteps
def get_alphas_cumprod_at_t(t):
    return alphas_cumprod.to(t.device)[t].view(-1, 1)

# Helper function for visualization
def plot_data(data, title):
    """Helper function to plot the 2D data."""
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    plt.title(title)
    plt.xlim(-2.5, 3.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Create a simple 2D dataset
X_0, _ = make_moons(n_samples=1000, noise=0.05)
X_0 = torch.tensor(X_0, dtype=torch.float32) * 1.5

print("Plotting original data...")
plot_data(X_0, "Original Data Distribution q(x_0)")


# ==============================================================================
# Part 1: The Forward Process (Noising)
# ==============================================================================

print("\n--- Part 1: The Forward Process ---")

def q_sample(x_0, t, noise=None):
    """
    Exercise 1.2: Sample x_t given x_0.
    x_0: a batch of original data points (e.g., images)
    t: a tensor of timesteps for each data point in the batch
    noise: optional standard Gaussian noise, otherwise will be generated
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # Get the \bar{\alpha}_t values for the given timesteps t
    alphas_t = get_alphas_cumprod_at_t(t)

    # YOUR CODE HERE: calculate x_t
    # Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    noisy_x = torch.sqrt(alphas_t) * x_0 + torch.sqrt(1 - alphas_t) * noise
    
    return noisy_x


print("Visualizing the forward process at different timesteps...")
plt.figure(figsize=(15, 3))
for i, t_val in enumerate([0, 50, 100, 500]):
    t = torch.full((X_0.shape[0],), t_val, dtype=torch.long)
    xt = q_sample(X_0, t)
    if xt is not None:
        plt.subplot(1, 4, i + 1)
        plt.scatter(xt[:, 0].numpy(), xt[:, 1].numpy(), alpha=0.5, s=10)
        plt.title(f't={t_val}')
        plt.xlim(-2.5, 3.5)
        plt.ylim(-2.5, 2.5)
plt.suptitle('Forward Process q(x_t | x_0)')
plt.show()


# ==============================================================================
# Part 3: The Loss Function and Training
# ==============================================================================

print("\n--- Part 3: The Loss Function and Training ---")

# A simple MLP to act as our denoiser
class DenoisingMLP(nn.Module):
    def __init__(self, data_dim=2, time_emb_dim=32):
        super().__init__()
        # Sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        # Main network that takes in data and time embedding
        self.main_net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, data_dim)
        )

    def forward(self, x, t):
        # x is (batch, data_dim)
        # t is (batch,)
        t_emb = self.get_time_embedding(t, 32)
        t_emb = self.time_mlp(t_emb)
        x_with_time = torch.cat((x, t_emb), dim=1)
        return self.main_net(x_with_time)
        
    def get_time_embedding(self, timesteps, embedding_dim):
        """Generates sinusoidal positional embeddings for timesteps."""
        assert embedding_dim % 2 == 0
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

# --- Training Loop ---
# Exercise 3.1: Fill in the training loop
model = DenoisingMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10001 # +1 to print the last epoch
batch_size = 1024

print("Starting training... (This may take a few minutes)")
for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    # 1. Sample a batch of data x_0
    indices = torch.randint(0, X_0.shape[0], (batch_size,))
    x_0 = X_0[indices]
    
    # 2. Sample random timesteps t
    t = torch.randint(0, T, (batch_size,), dtype=torch.long)
    
    # 3. Sample random noise ϵ from standard Gaussian distribution
    # YOUR CODE HERE
    noise = torch.randn_like(x_0)
    
    # 4. Create the noisy sample x_t using the q_sample function
    # YOUR CODE HERE
    x_t = q_sample(x_0, t, noise=noise)
    
    if isinstance(noise, str) or isinstance(x_t, str):
        if epoch == 0:
             print("Please fill in the blanks in the training loop!")
        continue

    # 5. Get the model's prediction of the noise
    predicted_noise = model(x_t, t)

    # 6. Calculate the loss with the MSE (mean square error) function
    # YOUR CODE HERE
    loss = torch.mean((predicted_noise - noise) ** 2)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# ==============================================================================
# Part 4: Sampling (Generating New Data)
# ==============================================================================

print("\n--- Part 4: Sampling ---")

@torch.no_grad()
def p_sample_loop(model, shape):
    """
    Exercise 3.2: Sample from the model by running the reverse process.
    """
    # Start from pure noise
    x = torch.randn(shape)
    
    for t_val in tqdm(reversed(range(0, T)), desc="Sampling", total=T):
        t = torch.full((shape[0],), t_val, dtype=torch.long)

        # Predict the noise using the model
        predicted_noise = model(x, t)
        
        # Get parameters for this timestep
        beta_t = betas[t_val]
        alpha_t = alphas[t_val]
        alpha_t_cumprod = alphas_cumprod[t_val]

        # The mean of the distribution for x_{t-1}
        # YOUR CODE HERE
        mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise)

        # Get the variance and add noise
        if t_val > 0:
            alpha_t_cumprod_prev = alphas_cumprod[t_val - 1]

            # The variance of the distribution for x_{t-1}
            # YOUR CODE HERE
            posterior_variance = beta_t * (1 - alpha_t_cumprod_prev) / (1 - alpha_t_cumprod)

            noise = torch.randn_like(x) * torch.sqrt(posterior_variance)
        else: # No noise at the last step
            noise = 0

        x = mean + noise
        
    return x


print("Generating new samples...")
model.eval()
generated_samples = p_sample_loop(model, shape=(1000, 2))

if generated_samples is not None:
    print("Plotting generated samples vs. original data...")
    plot_data(generated_samples.detach().cpu().numpy(), "Generated Samples")
    plot_data(X_0, "Original Data Distribution")
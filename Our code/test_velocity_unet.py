import torch
from model import VelocityUNet, RectifiedFlow

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate network and wrapper
net = VelocityUNet().to(device)
rf = RectifiedFlow(net)

# Dummy batch of 4 grayscale images, 28x28
x_dummy = torch.randn(4, 1, 28, 28, device=device)

# Dummy time tensor
t_dummy = torch.rand(4, device=device)

# Forward pass through VelocityUNet
v_pred = net(x_dummy, t_dummy)
print("VelocityUNet output shape:", v_pred.shape)  # should be (4, 1, 28, 28)

# Compute training loss
loss = rf.get_loss(x_dummy)
print("RectifiedFlow training loss:", loss.item())

# Test sampling
schedule = torch.linspace(0, 1, 5)  # 5-step schedule
samples = rf.sample(n=2, schedule=schedule, device=device)
print("Sampled images shape:", samples.shape)  # should be (2, 1, 28, 28)

# Optional: compute flow matching error at t=0.5
fme = rf.compute_flow_matching_error(x_dummy, t=0.5, n_samples=2)
print("Flow matching error:", fme)
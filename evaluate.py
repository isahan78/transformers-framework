# This code defines an evaluation loop, which loads the trained model, moves the data to GPU, generates the src and tgt masks,
#  passes the data through the model, and computes the perplexity. It also prints the perplexity on the screen.


import torch
from torch.utils.data import DataLoader
from model import Transformer

# Instantiate the model
model = Transformer(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

# Load the trained model parameters
model.load_state_dict(torch.load("trained_model.pt"))

# Move the model to the evaluation mode
model.eval()

# Define the dataset and data loader for evaluation
dataset = ... # Some dataset object
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Evaluation loop
with torch.no_grad():
    for i, (src, tgt) in enumerate(data_loader):
        # Move the data to the GPU
        src = src.to(device)
        tgt = tgt.to(device)

        # Generate the src and tgt masks
        src_mask, tgt_mask = ...

        # Forward pass
        output = model(src, tgt, src_mask, tgt_mask)

        # Compute the perplexity
        prob = output.view(-1, output.size(-1)).softmax(dim=-1)
        log_prob = torch.log(prob)
        tgt = tgt.view(-1)
        non_pad_mask = tgt != PAD_TOKEN
        perplexity = torch.exp(-log_prob.gather(1, tgt.view(-1, 1))[non_pad_mask].mean())
        
        # Print the perplexity
        print(f'Perplexity : {perplexity.item():.4f}')

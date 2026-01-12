import torch
import torch.nn as nn

class BDH_Reasoner(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, sparsity=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        
        # Synaptic Matrix (The "Brain")
        self.synapses = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        
        # Persistent State (Internal Belief)
        self.state = torch.zeros(1, hidden_dim)

    def apply_hebbian_update(self, activations):
        """Core BDH Principle: Neurons that fire together, wire together."""
        with torch.no_grad():
            # Calculate outer product for synaptic reinforcement
            update = torch.matmul(activations.T, activations) 
            self.synapses.data += 0.001 * update 
            # Normalize to prevent exploding gradients
            self.synapses.data /= self.synapses.data.norm()

    def forward(self, x, train_synapses=True):
        # 1. Encode input to hidden space
        h = torch.relu(self.encoder(x))
        
        # 2. Sparse Competition (Only top neurons fire)
        k = int(self.hidden_dim * self.sparsity)
        top_k_values, _ = torch.topk(h, k)
        min_val = top_k_values[:, -1]
        h_sparse = torch.where(h >= min_val, h, torch.zeros_like(h))
        
        # 3. Persistent State Integration
        self.state = torch.tanh(torch.matmul(h_sparse, self.synapses) + self.state)
        
        # 4. Learning step
        if train_synapses:
            self.apply_hebbian_update(h_sparse)
            
        return self.classifier(self.state)

    def reset_state(self):
        self.state = torch.zeros(1, self.hidden_dim)

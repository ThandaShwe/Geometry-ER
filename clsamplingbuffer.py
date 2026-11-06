import torch
import torch.nn as nn
import numpy as np

class MHAM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True
        )
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        h = self.proj(x)
        attn_output, attn_weights = self.attn(h, h, h)
        scores = attn_weights.mean(dim=1)  # shape: (batch,)
        return scores

class CLSamplingBuffer:
    def __init__(self, max_size, input_dim, device):
        self.max_size = max_size
        self.device = device
        self.mham = MHAM(input_dim).to(device)
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_task = []

    def add_batch(self, model, x, y, task_id):
        for i in range(len(x)):
            self.buffer_x.append(x[i])
            self.buffer_y.append(y[i])
            self.buffer_task.append(task_id)

        if len(self.buffer_x) > self.max_size:
            self._prune(model)

    def _prune(self, model):
        X = torch.tensor(np.array(self.buffer_x), dtype=torch.float32).to(self.device)
        scores = self.mham(X).detach().cpu().numpy()
        idx = np.argsort(scores)[-self.max_size:]

        self.buffer_x = [self.buffer_x[i] for i in idx]
        self.buffer_y = [self.buffer_y[i] for i in idx]
        self.buffer_task = [self.buffer_task[i] for i in idx]

    def sample(self, batch_size, model):
        if len(self.buffer_x) == 0:
            return None, None

        X = torch.tensor(np.array(self.buffer_x), dtype=torch.float32).to(self.device)
        scores = self.mham(X).detach().cpu().numpy()
        idx = np.argsort(scores)[-batch_size:]

        replay_x = np.array([self.buffer_x[i] for i in idx])
        replay_y = np.array([self.buffer_y[i] for i in idx])

        return replay_x, replay_y

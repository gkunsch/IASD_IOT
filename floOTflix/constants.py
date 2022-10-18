import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42 # Don't need to be an arg, only to be fixed
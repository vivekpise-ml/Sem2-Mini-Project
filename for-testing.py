import torch
sd = torch.load("models/charcnn_model.pt", map_location="cpu")
print(type(sd))
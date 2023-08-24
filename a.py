import torch

prototype = torch.load('output_prototype/prototype.pth',map_location='cpu')
print(prototype)
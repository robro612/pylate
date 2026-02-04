import torch

x = torch.tensor([1.0, 2.0, 3.0])
mask = torch.tensor([True, False, True])

print("Original:", x)

x.masked_fill(mask, 0)
print("After masked_fill (no underscore):", x)

x.masked_fill_(mask, 0)
print("After masked_fill_ (with underscore):", x)

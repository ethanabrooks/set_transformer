import math

import torch
from tqdm import tqdm


def quantize_tensor(tensor, n_bins):
    # Flatten tensor
    flat_tensor = tensor.flatten()

    # Sort the flattened tensor
    sorted_tensor, _ = torch.sort(flat_tensor)

    # Determine the thresholds for each bin
    n_points_per_bin = int(math.ceil(len(sorted_tensor) / n_bins))
    thresholds = sorted_tensor[::n_points_per_bin].contiguous()

    # Assign each value in the flattened tensor to a bucket
    # The bucket number is the quantized value
    quantized_tensor = torch.bucketize(flat_tensor, thresholds)

    # Make the quantized values contiguous
    unique_bins = torch.unique(quantized_tensor)
    for i, bin in enumerate(unique_bins):
        quantized_tensor[quantized_tensor == bin] = i

    # Reshape the quantized tensor to the original tensor's shape
    quantized_tensor = quantized_tensor.view(tensor.shape)

    return quantized_tensor


def round_tensor(tensor: torch.Tensor, round_to: int, contiguous: bool = False):
    discretized = (tensor * round_to).round().long()

    if contiguous:
        # Make the quantized values contiguous
        unique_bins = torch.unique(discretized)
        for i, bin in enumerate(tqdm(unique_bins, desc="Contiguous")):
            discretized[discretized == bin] = i

    # Reshape the quantized tensor to the original tensor's shape
    discretized = discretized.view(tensor.shape)

    return discretized


if __name__ == "__main__":
    quantize_tensor  # whitelist

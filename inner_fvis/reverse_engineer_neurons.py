import torch
from utils.imagenet_dataset import ImageNet
import torchvision.models as models
import os
from math import ceil
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import argparse

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument('--target_layer', type=str, default='layer4_2')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--chs', type=list_of_ints, default="54,1935")
parser.add_argument('--split_N', type=int, default=None)

args = parser.parse_args()
print(args)

def split_network(model, req_name):
    """
    Splits a PyTorch model into a feature extractor up to the specified layer.

    Args:
        model (torch.nn.Module): The PyTorch model to split.
        req_name (str): The name of the layer up to which the feature extractor is created.

    Returns:
        torch.nn.Sequential: A sequential model containing layers up to the specified layer.
    """
    layers = []
    feat_ext = []

    def get_layers(net, prefix=[]):
        """
        Recursively traverses the model to collect layers up to the specified layer.

        Args:
            net (torch.nn.Module): The current module being traversed.
            prefix (list): The prefix list to track the hierarchical layer names.
        """
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layers:
                    if layers[-1] == req_name:
                        break
                if layer is None:
                    # Skip layers like GoogLeNet's aux1 and aux2
                    continue
                if layer.__class__.__name__ == 'Sequential':
                    get_layers(layer, prefix=prefix + [name])
                else:
                    layers.append("_".join(prefix + [name]))
                    feat_ext.append(layer)

    get_layers(model)
    return torch.nn.Sequential(*feat_ext)


def generate_dataloader(batch_size=64, split_N=None):
    """
    Generates a PyTorch DataLoader for the ImageNet dataset.

    Args:
        batch_size (int): The number of samples per batch to load. Default is 64.
        split_N (int, optional): If specified, loads a subset of the dataset with split_N samples.

    Returns:
        torch.utils.data.DataLoader: A DataLoader object for the ImageNet dataset.
    """
    if split_N is None:
        dataset_real = ImageNet(split='train')
    else:
        dataset_real = ImageNet(split='train_subset', split_N=split_N)

    dloader_real = torch.utils.data.DataLoader(
        dataset_real,
        batch_size=batch_size,  # Adjust based on GPU memory
        num_workers=4,          # Adjust based on available CPUs and RAM
        shuffle=False,
        drop_last=False,
        pin_memory=False
    )
    
    return dloader_real

def patchify(inputs, patch_size=64):
    """
    Extracts overlapping patches from the input tensor.

    Args:
        inputs (torch.Tensor): The input tensor of shape (N, C, H, W), where
                               N is the batch size,
                               C is the number of channels,
                               H and W are the height and width of the input images.
        patch_size (int): The size of each patch (patch_size x patch_size). Default is 64.

    Returns:
        torch.Tensor: A tensor containing the extracted patches of shape 
                      (num_patches, C, patch_size, patch_size), where
                      num_patches = N * (H' * W') and
                      H' and W' are the number of patches along the height and width dimensions.
    """
    strides = int(patch_size * 0.80)

    patches = torch.nn.functional.unfold(inputs, kernel_size=patch_size, stride=strides)
    patches = patches.transpose(1, 2).contiguous().view(-1, inputs.shape[1], patch_size, patch_size)
    return patches

def generate_model(arch='resnet50', target='layer4_2'):
    """
    Generates a PyTorch model truncated at the specified target layer.

    Args:
        arch (str): The architecture of the model to load (e.g., 'resnet50').
        target (str): The name of the layer up to which the model is truncated.

    Returns:
        torch.nn.Sequential: A PyTorch model truncated at the specified target layer.
    """
    net = models.__dict__[arch](pretrained=True)  # Load the specified model architecture
    net = net.to('cuda')  # Move the model to GPU
    print('==> Resuming from checkpoint..')
    net = net.eval()  # Set the model to evaluation mode

    # Truncate the model at the specified target layer
    net = split_network(model=net, req_name=target)

    return net

def find_repeated_indices(lst):
    """
    Finds the indices of repeated elements in a list, excluding the first occurrence.

    Args:
        lst (list): The input list to check for repeated elements.

    Returns:
        list: A list of indices corresponding to repeated elements in the input list.
    """
    # Create a dictionary to store the first occurrence of elements
    first_occurrence = {}
    
    # Create a list to store the indices of repeated elements (excluding first encounters)
    repeated_indices = []
    
    # Iterate through the list and collect repeated indices (excluding the first occurrence)
    for i, elem in enumerate(lst):
        if elem in first_occurrence:
            # If the element has already been encountered, add the current index to repeated_indices
            repeated_indices.append(i)
        else:
            # Store the first occurrence of the element
            first_occurrence[elem] = i
    
    return repeated_indices

def remove_indices(data, indices_to_remove):
    """
    Removes elements from the input data at the specified indices.

    Args:
        data (torch.Tensor or list): The input data from which elements are to be removed.
                                     Can be a PyTorch tensor or a Python list.
        indices_to_remove (list or torch.Tensor): The indices of elements to remove.

    Returns:
        torch.Tensor or list: The input data with elements at the specified indices removed.
                              The return type matches the type of the input data.

    Raises:
        ValueError: If the input data is neither a list nor a PyTorch tensor.
    """
    # Check if data is a torch tensor or a list
    if isinstance(data, torch.Tensor):
        # Convert indices to a tensor for torch operations
        indices_to_remove = torch.tensor(indices_to_remove)
        
        # Create a mask with True for elements to keep and False for elements to remove
        mask = torch.ones(data.size(0), dtype=torch.bool)
        mask[indices_to_remove] = False
        
        # Return the tensor with elements at the specified indices removed
        return data[mask]
    
    elif isinstance(data, list):
        # Remove elements from the list by filtering out the indices to remove
        return [elem for i, elem in enumerate(data) if i not in indices_to_remove]
    
    else:
        raise ValueError("Input data must be either a list or a PyTorch tensor.")

def extract_features(dataloader, net, patch_size, channels):
    """
    Extracts and saves the top patches for specified channels from the input data.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader object for the dataset.
        net (torch.nn.Module): The PyTorch model used for feature extraction.
        patch_size (int): The size of the patches to extract.
        channels (list): List of channel indices for which to extract features.

    Returns:
        None: Saves the top patches and their corresponding file paths to disk.
    """
    nb_batch_patch = 50

    max_patches = [[]] * len(channels)
    max_values = [[]] * len(channels)

    combined_val = [[]] * len(channels)
    combined_patches = [[]] * len(channels)

    first_encounter = torch.ones((len(channels)))
    max_samples = [[]] * len(channels)

    for i, (img_real, target_real) in enumerate(tqdm(dataloader, disable=False), 0):

        with torch.no_grad():
            img_real = img_real.to('cuda')
            batch_size = img_real.shape[0]

            patches = patchify(inputs=img_real, patch_size=patch_size)

            num_patches = int(patches.shape[0] / img_real.shape[0])

            samples = [item[0] for item in dataloader.dataset.data.samples[i * batch_size:i * batch_size + batch_size]]

            nb_batchs = ceil(len(patches) / nb_batch_patch)

            start_ids = [b * nb_batch_patch for b in range(nb_batchs)]

            for j in start_ids:

                x = torch.tensor(patches[j:j + nb_batch_patch])

                indices = np.array(range(j, j + nb_batch_patch))
                indices = indices[:x.shape[0]]
                ind_img = np.array(np.floor(indices / num_patches), dtype=int)

                x_resized = torch.nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)

                patch_outputs = net(x_resized).mean((2, 3))

                for ch in range(len(channels)):

                    if first_encounter[ch]:

                        max_patches[ch] = x.clone()
                        max_samples[ch] = list(map(samples.__getitem__, ind_img))
                        max_values[ch] = patch_outputs[:, channels[ch]]
                        first_encounter[ch] = 0

                    else:

                        combined_val[ch] = torch.cat((max_values[ch], patch_outputs[:, channels[ch]]))
                        combined_patches[ch] = torch.cat((max_patches[ch], x))
                        max_samples[ch] += list(map(samples.__getitem__, ind_img))
                        _, gathered_ind = torch.sort(combined_val[ch], descending=True)

                        max_values[ch] = combined_val[ch][gathered_ind[:nb_batch_patch]]
                        max_patches[ch] = combined_patches[ch][gathered_ind[:nb_batch_patch]]
                        max_samples[ch] = list(map(max_samples[ch].__getitem__, gathered_ind[:nb_batch_patch]))

                        repeated_files_indices = find_repeated_indices(max_samples[ch])
                        if len(repeated_files_indices) > 0:

                            max_values[ch] = remove_indices(max_values[ch], repeated_files_indices)
                            max_patches[ch] = remove_indices(max_patches[ch], repeated_files_indices)
                            max_samples[ch] = remove_indices(max_samples[ch], repeated_files_indices)

    for ch in range(len(channels)):

        dir_save = f'{args.arch}/neuron_{target_layer}/{channels[ch]}/top{nb_batch_patch}/'
        txt_filename = dir_save + f'topk_files.txt'
        os.makedirs(dir_save, exist_ok=True)
        vutils.save_image(max_patches[ch], dir_save + f'patches.png', normalize=True, scale_each=True, nrow=int(5))
        with open(txt_filename, 'w') as f:
            for id in range(nb_batch_patch):
                f.write(f"{max_samples[ch][id]}\n")
                    
if __name__ == '__main__':

    batch_size = args.batch_size
    arch = args.arch
    target_layer = args.target_layer
    chs = args.chs
    split_N = args.split_N
    patch_size = args.patch_size

    # Generate a DataLoader for the ImageNet dataset
    dataloader = generate_dataloader(batch_size=batch_size, split_N=split_N)
    
    # Generate a PyTorch model truncated at the specified target layer
    net = generate_model(arch=arch, target=target_layer)

    # Extract and save the top patches for the specified channels
    extract_features(dataloader=dataloader,
                     net=net,
                     patch_size=patch_size,
                     channels=chs)
    

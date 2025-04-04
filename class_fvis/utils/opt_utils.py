from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import torch
from PIL import Image
import numpy as np
import os
from utils.utils import lr_cosine_policy, clip, denormalize
import torchvision.transforms as Tr
from torchvision.models.feature_extraction import create_feature_extractor

IMAGENET_DIR = "/scratch/inf0/user/mparcham/ILSVRC2012/train/" # Imagenet directory

def sort_matching(target, input):
    """
    Computes the mean squared error (MSE) loss between a target tensor and an input tensor 
    after sorting and matching their values.
    This function is designed to compare a target tensor and an input tensor by sorting 
    their values and aligning them based on the sorted indices for distribution matching. 
    Args:
        target (torch.Tensor): A 4D tensor of shape (B, C, H, W), where B is the batch size, 
                               C is the number of channels, H is the height, and W is the width 
                               of the spatial dimensions.
        input (torch.Tensor): A 4D tensor of shape (B, C, H, W), where B is the batch size, 
                              C is the number of channels, H is the height, and W is the width 
                              of the spatial dimensions.
    Returns:
        torch.Tensor: A scalar tensor representing the mean squared error (MSE) loss 
                      between the sorted and matched values of the input and target tensors.
    """
    B, C = target.size(0), target.size(1)

    # Compute the mean across the batch dimension for the input tensor
    input_mean = input.mean(0)  # Shape: (C, H, W)

    # Flatten the spatial dimensions and sort the input tensor
    _, index_content = torch.sort(input_mean.view(C, -1), dim=-1)
    inverse_index = index_content.argsort(dim=-1)

    # Sort the target tensor along the spatial dimensions
    value_style, _ = torch.sort(target.view(B, C, -1), dim=-1)
    value_style = value_style.mean(0)  # Shape: (C, H*W)

    # Compute the MSE loss between the sorted and matched values
    mse_loss = (input_mean.view(C, -1) - value_style.gather(-1, inverse_index)) ** 2

    return mse_loss.mean()

def get_image_prior_losses(inputs_jit):
    """
    Computes image prior losses based on total variation regularization.

    This function calculates two types of losses for the input tensor:
    1. L2 norm-based total variation loss (`loss_var_l2`), which measures the smoothness of the image.
    2. L1 norm-based total variation loss (`loss_var_l1`), which is scaled and normalized to account for pixel intensity range.

    Args:
        inputs_jit (torch.Tensor): A 4D tensor of shape (B, C, H, W), where:
            - B is the batch size,
            - C is the number of channels,
            - H is the height of the image,
            - W is the width of the image.
          This tensor represents the input image or batch of images.

    Returns:
        tuple: A tuple containing:
            - loss_var_l1 (torch.Tensor): The L1 norm-based total variation loss.
            - loss_var_l2 (torch.Tensor): The L2 norm-based total variation loss.
    """

    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def obtain_real_imgs(targets, num_real_img):
    """
    Loads and preprocesses a specified number of real images from the ImageNet dataset for a given class.

    Args:
        targets (int): The class index of the ImageNet dataset to load images from.
        num_real_img (int): The number of real images to load and preprocess.

    Returns:
        torch.Tensor: A 4D tensor of shape (num_real_img, C, H, W), where:
            - C is the number of channels (3 for RGB),
            - H and W are the height and width of the images (224x224 after preprocessing).
    """
    normalize = Tr.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    trans = [
        Tr.Resize(256),
        Tr.CenterCrop(224),
        Tr.PILToTensor(),
        Tr.ConvertImageDtype(torch.float),
        normalize,
    ]

    transforms_all = Tr.Compose(trans)

    # Get the class path for the specified target class
    class_number = targets  # ImageNet class index
    class_path = sorted(os.listdir(IMAGENET_DIR))[class_number]
    class_files = os.listdir(os.path.join(IMAGENET_DIR, class_path))

    # Shuffle the files to ensure randomness
    random.shuffle(class_files)

    img_real = []
    inc_img = 0

    # Load and preprocess images
    for file in class_files:
        if inc_img == num_real_img:
            break
        img = Image.open(os.path.join(IMAGENET_DIR, class_path, file))
        num_channel = len(img.split())
        if num_channel != 3:  # Skip non-RGB images
            continue
        else:
            inc_img += 1
        img_p = transforms_all(img)
        img_real.append(img_p)

    # Stack and move images to GPU
    img_real = torch.stack(img_real, 0)
    img_real = img_real.to('cuda')
    return img_real


class DeepFeaturesClass(object):
    def __init__(self,
                 model=None,
                 parameters=dict(),
                 coefficients=dict(),
                 exp_name=None,
                 folder_name=None):

        self.model = model

        self.image_resolution = parameters["resolution"]
        self.do_flip = parameters["do_flip"]
        self.setting_id = parameters["setting_id"]
        self.bs = parameters["bs"]  # batch size
        self.jitter = parameters["jitter"]
        self.num_real_img = parameters["num_real_img"]
        self.epochs = parameters['epochs']
        self.arch_name = parameters['arch_name']

        self.print_every = 100
 
        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.feat_dist = coefficients["feat_dist"]
        self.layer_weights = coefficients["layer_weights"]

        self.num_generations = 0

        # Create folders for images and logs
        self.exp_name = exp_name
        self.folder_name = folder_name

    def get_images(self, targets=None):
        print("get_images call")

        model = self.model
        img_original = self.image_resolution
        print_every = self.print_every
        data_type = torch.float
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        # Define the return nodes for different architectures for feature extraction
        if 'resnet' in self.arch_name:
            return_nodes = {
                            "conv1": "conv1",
                            "layer1": "layer1",
                            "layer2": "layer2",
                            "layer3": "layer3",
                            "layer4": "layer4",
                            "fc": "fc"
                        }
        elif 'vit_b' in self.arch_name:
            return_nodes = {
                            "conv_proj": "conv1",
                            "encoder.layers.encoder_layer_0": "layer1",
                            "encoder.layers.encoder_layer_4": "layer2",
                            "encoder.layers.encoder_layer_8": "layer3",
                            "encoder.layers.encoder_layer_11": "layer4",
                            "heads.head": "fc"
                        }
        elif 'vit_l' in self.arch_name:
            return_nodes = {
                            "conv_proj": "conv1",
                            "encoder.layers.encoder_layer_2": "layer1",
                            "encoder.layers.encoder_layer_9": "layer2",
                            "encoder.layers.encoder_layer_20": "layer3",
                            "encoder.layers.encoder_layer_23": "layer4",
                            "heads.head": "fc"
                        }
        elif 'densenet' in self.arch_name:
            return_nodes = {
                            "features.conv0": "conv1",
                            "features.denseblock1": "layer1",
                            "features.denseblock2": "layer2",
                            "features.denseblock3": "layer3",
                            "features.denseblock4": "layer4",
                            "classifier": "fc"
                        }
        elif 'convnext' in self.arch_name:
            return_nodes = {
                            "features.0": "conv1",
                            "features.2.1": "layer1",
                            "features.4.1": "layer2",
                            "features.6.1": "layer3",
                            "features.7.2.add": "layer4",
                            "classifier.2": "fc"
                        }

        # Create feature extractor model
        model2 = create_feature_extractor(model, return_nodes=return_nodes)
        model2.eval()

        # Load real images
        img_real = obtain_real_imgs(targets=targets,
                                    num_real_img=self.num_real_img)

        # Initialize the syntethic image
        inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device='cuda',
                             dtype=data_type)
        
        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else self.epochs
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal resolution
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif self.setting_id == 1:
                #2k normal resolultion
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif self.setting_id == 2:
                #20k normal resolution
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps = 1e-8)
                do_clip = False

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                    img_real_jit = pooling_function(img_real)
                else:
                    inputs_jit = inputs
                    img_real_jit = img_real

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()

                # get synthetic image features
                syn_out = model2(inputs_jit)
                
                # get real image features
                if iteration == 1:
                    
                    with torch.no_grad():
                        real_out = model2(img_real_jit)

                # perform distribution matching through sort-matching loss        
                if 'vit' in self.arch_name:
                    loss_conv1 = sort_matching(input=syn_out["conv1"],
                                              target=real_out["conv1"]) 
                    
                    loss_layer1 = sort_matching(input=syn_out["layer1"].permute(0,2,1),
                                               target=real_out["layer1"].permute(0,2,1))
                    
                    loss_layer2 = sort_matching(input=syn_out["layer2"].permute(0,2,1),
                                               target=real_out["layer2"].permute(0,2,1))
                    
                    loss_layer3 = sort_matching(input=syn_out["layer3"].permute(0,2,1),
                                               target=real_out["layer3"].permute(0,2,1))
                    
                    loss_layer4 = sort_matching(input=syn_out["layer4"].permute(0,2,1),
                                               target=real_out["layer4"].permute(0,2,1))
                     
                else:                
                    loss_conv1 = sort_matching(input=syn_out["conv1"],
                                              target=real_out["conv1"]) 
                    
                    loss_layer1 = sort_matching(input=syn_out["layer1"],
                                               target=real_out["layer1"])
                    
                    loss_layer2 = sort_matching(input=syn_out["layer2"],
                                               target=real_out["layer2"])
                    
                    loss_layer3 = sort_matching(input=syn_out["layer3"],
                                               target=real_out["layer3"])
                    
                    loss_layer4 = sort_matching(input=syn_out["layer4"],
                                               target=real_out["layer4"])
                    
                # total distribution matching loss
                loss_add = self.layer_weights[0]*loss_conv1 +\
                        self.layer_weights[4]*loss_layer4 +\
                        self.layer_weights[3]*loss_layer3 +\
                        self.layer_weights[1]*loss_layer1 +\
                        self.layer_weights[2]*loss_layer2 
                
                # image prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                # combining losses
                loss = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.l2_scale * loss_l2 + \
                           self.feat_dist * loss_add

                if iteration % print_every==0:
                    print("------------iteration {}----------".format(iteration))
                    print("total loss", loss.item())

                loss.backward()
                optimizer.step()

                # clip color outlayers
                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=False)

        # save the image
        best_inputs = inputs.data.clone()
        best_inputs = denormalize(best_inputs)
        self.save_images(best_inputs)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images):
        """
        Saves the generated images locally.

        Args:
            images (torch.Tensor): A 4D tensor of shape (B, C, H, W), where:
                - B is the batch size,
                - C is the number of channels (3 for RGB),
                - H and W are the height and width of the images.
        """
        for id in range(images.shape[0]):
            if self.exp_name is None:
                name = f"c1_{self.layer_weights[0]}_l1_{self.layer_weights[1]}_l2_{self.layer_weights[2]}_l3_{self.layer_weights[3]}_l4_{self.layer_weights[4]}.png"
                place_to_store = os.path.join(self.folder_name, name)
            else:
                place_to_store = os.path.join(self.folder_name, self.exp_name + f"_{id}.png")

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)
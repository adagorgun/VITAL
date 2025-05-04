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
from utils.lrp import LRPModel, LRPModelRestricted
import copy
from utils.guided_backprop import LayerGuidedBackprop

def split_network(model, req_name):
    """
    Splits the given model into a feature extractor up to the specified layer.

    Args:
        model (torch.nn.Module): The neural network model to split.
        req_name (str): The name of the layer up to which the feature extractor is created.

    Returns:
        torch.nn.Sequential: A sequential model containing the layers up to the specified layer.
    """
    layers = []
    feat_ext = []

    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layers:
                    if layers[-1] == req_name:
                        break
                if layer is None:
                    # Skip None layers (e.g., GoogLeNet's aux1 and aux2 layers)
                    continue
                if layer.__class__.__name__ == 'Sequential':
                    get_layers(layer, prefix=prefix + [name])
                else:
                    layers.append("_".join(prefix + [name]))
                    feat_ext.append(layer)

    get_layers(model)
    return torch.nn.Sequential(*feat_ext)
    
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

    B, C, W, H = target.size()

    # Remove the batch dimension from the input tensor
    input = input.squeeze(0)
    
    # Flatten the spatial dimensions and sort the input tensor
    _, index_content = torch.sort(input.view(C, -1))
    inverse_index = index_content.argsort(-1)

    # Sort the target tensor along the spatial dimensions
    value_style, _ = torch.sort(target.view(B, C, -1))
    value_style = value_style.mean(0)

    # Compute the MSE loss between the sorted and matched values
    mse_loss = (input.view(C, -1) - value_style.gather(-1, inverse_index))**2

    return mse_loss.unsqueeze(0)

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


def obtain_real_imgs(channel, num_real_img, main_dir):
    """
    Loads and preprocesses real images for a specific channel.

    Args:
        channel (int): The channel index for which images are to be obtained.
        num_real_img (int): The number of real images to load.
        main_dir (str): The folder of the files.txt

    Returns:
        torch.Tensor: A tensor containing the preprocessed real images.
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

    txt_file = os.path.join(main_dir, str(channel),'topk_files.txt') 

    with open(txt_file) as f:
        datafiles = [line.rstrip() for line in f]

    datafiles = list(dict.fromkeys(datafiles))

    img_real = []
    inc_img = 0

    for file in datafiles:
        if inc_img == num_real_img:
            break
        img = Image.open(file)
        num_channel = len(img.split())
        if num_channel != 3:
            continue
        else:
            inc_img += 1
        img_p = transforms_all(img)
        img_real.append(img_p)

    img_real = torch.stack(img_real, 0)
    img_real = img_real.to('cuda')
    return img_real

def to_numpy(tensor):
    # Ensure tensor is on CPU and convert to NumPy
    return tensor.detach().cpu().numpy()

def check_format(arr):
    # ensure numpy array and move channels to the last dimension
    # if they are in the first dimension
    if isinstance(arr, torch.Tensor):
        arr = to_numpy(arr)
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    return arr

def normalize(image):
    # normalize image to 0-1 range
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= (image.max()+1e-12)
    return image

def clip_percentile(img, percentile=0.1):
    # clip pixel values to specified percentile range
    return np.clip(img, np.percentile(img, percentile), np.percentile(img, 100-percentile))

def save_maco(image, alpha, percentile_image=1.0, percentile_alpha=95, filename='image.png'):
    # visualize image with alpha mask overlay after normalization and clipping
    image, alpha = check_format(image), check_format(alpha)
    image = clip_percentile(image, percentile_image)
    image = normalize(image)

    # mean of alpha across channels, clipping, and normalization
    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    image_np = np.concatenate([image, alpha], -1)

    # overlay alpha mask on the image
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    pil_image.save(filename) 

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
        self.bs = parameters["bs"]  
        self.jitter = parameters["jitter"]
        self.num_real_img = parameters["num_real_img"]
        self.epochs = parameters['epochs']
        self.channel = parameters["channel"]
        self.layer = parameters["layer"]
        self.arch_name = parameters['arch_name']
        self.method = parameters["method"]
        self.topk_dir = parameters["topk_dir"]

        self.print_every = 100
 
        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.feat_dist = coefficients["feat_dist"]
        self.layer_weights = coefficients["layer_weights"]

        self.num_generations = 0

        ## Create folders for images and logs
        self.exp_name = exp_name
        self.folder_name = folder_name

    def get_images(self):
        print("get_images call")

        model = self.model
        model_split = split_network(copy.deepcopy(model), self.layer)
        model_split.eval()

        # LRP
        skip_connection_prop = "flows_skip"
        rel_pass_ratio=1.0

        print_every = self.print_every
        img_original = self.image_resolution
        data_type = torch.float
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        # define the return nodes for the feature extractor and LRP model
        if 'resnet' in self.arch_name:
            return_nodes = {
                            "conv1": "conv1",
                            "layer1": "layer1",
                            "layer2": "layer2",
                            "layer3": "layer3",
                            "layer4": "layer4",
                            "fc": "fc"
                        }
            
            return_nodes_lrp = {"fdim": [1024, 512, 256, 64],
                                "res": [14, 28, 56, 112]}
            
            target_layers = {
            "layer3": model_split[16],
            "layer2": model_split[10],
            "layer1": model_split[6],
            "conv1" : model_split[0]
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
            return_nodes_lrp = {"fdim": [1024, 512, 256, 64],
                                "res": [14, 28, 56, 112]}
            
        else:
            raise Exception("Not Implemented Yet!")

        # create the feature extractor and LRP model
        model2 = create_feature_extractor(copy.deepcopy(model), return_nodes=return_nodes)
        model2.eval()

        # obtain real images for the specified channel, layer, and architecture
        img_real = obtain_real_imgs(channel=self.channel,
                                    num_real_img=self.num_real_img, 
                                    main_dir=self.topk_dir)
        
        # create the input tensor and transparency accumulator
        inputs = torch.randn((self.bs, 3, img_original, img_original), requires_grad=True, device='cuda',
                             dtype=data_type)
                        
        transparency_accumulator = torch.zeros((self.bs, 3, img_original, img_original), device='cuda',
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
                    img_real_jit = img_real
                    inputs_jit = inputs

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

                img_real_jit.requires_grad = True

                # get synthetic image features
                syn_out = model2(inputs_jit)
                
                # get real image features and relevance maps
                if iteration==1:
                    real_out = model2(img_real_jit)

                    if self.method == 'LRP':
                        lrp_model_real = LRPModel(model_split, rel_pass_ratio=rel_pass_ratio, skip_connection_prop=skip_connection_prop)
                        attention_real = lrp_model_real.forward(img_real_jit, channel=self.channel, return_nodes=copy.deepcopy(return_nodes_lrp))
                    elif self.method == 'LRPRestricted':
                        lrp_model_real = LRPModelRestricted(model_split, rel_pass_ratio=rel_pass_ratio, skip_connection_prop=skip_connection_prop)
                        attention_real = lrp_model_real.forward(img_real_jit, channel=self.channel, return_nodes=copy.deepcopy(return_nodes_lrp))
                    elif self.method == 'GuidedBackprop':
                        assert "resnet" in self.arch_name # currently only supported for resnet
                        guided_bp = LayerGuidedBackprop(model_split)
                        attention_real = guided_bp.attribute(inputs=img_real_jit, target_layers=target_layers, target=self.channel)
                    else:
                        raise ValueError("Method not supported")
                    
                    importance_neuron = []

                    for att_id in range(len(attention_real)):

                        imp_neuron = attention_real[att_id].amax((2,3))
                        imp_neuron = imp_neuron - imp_neuron.amin(1, keepdim=True)
                        imp_neuron = imp_neuron / (imp_neuron.amax(1, keepdim=True) + 1e-6) 

                        importance_neuron.append(imp_neuron)

                        attention_real[att_id] = attention_real[att_id] - attention_real[att_id].amin((2,3), keepdim=True)
                        attention_real[att_id] = attention_real[att_id] / (attention_real[att_id].amax((2,3), keepdim=True) + 1e-6) 

                    attention_conv1 = attention_real[3]*real_out["conv1"]
                    attention_layer1 = attention_real[2]*real_out["layer1"]
                    attention_layer2 = attention_real[1]*real_out["layer2"]
                    attention_layer3 = attention_real[0]*real_out["layer3"]

                # get synthetic relevance maps
                if self.method == 'LRP':
                    lrp_model_syn= LRPModel(model_split, rel_pass_ratio=rel_pass_ratio, skip_connection_prop=skip_connection_prop)
                    attention_syn = lrp_model_syn.forward(inputs_jit, channel=self.channel, return_nodes=copy.deepcopy(return_nodes_lrp))
                elif self.method == 'LRPRestricted':
                    lrp_model_syn= LRPModelRestricted(model_split, rel_pass_ratio=rel_pass_ratio, skip_connection_prop=skip_connection_prop)
                    attention_syn = lrp_model_syn.forward(inputs_jit, channel=self.channel, return_nodes=copy.deepcopy(return_nodes_lrp))
                elif self.method == 'GuidedBackprop':
                    assert "resnet" in self.arch_name # currently only supported for resnet
                    guided_bp = LayerGuidedBackprop(model_split)
                    attention_syn = guided_bp.attribute(inputs=inputs_jit, target_layers=target_layers, target=self.channel)
                else:
                    raise ValueError("Method not supported")

                for att_id in range(len(attention_syn)):
                    attention_syn[att_id] = attention_syn[att_id] - attention_syn[att_id].amin((2,3), keepdim=True)
                    attention_syn[att_id] = attention_syn[att_id] / (attention_syn[att_id].amax((2,3), keepdim=True) + 1e-6) 

                attention_syn_conv1 = attention_syn[3]*syn_out["conv1"]
                attention_syn_layer1 = attention_syn[2]*syn_out["layer1"]
                attention_syn_layer2 = attention_syn[1]*syn_out["layer2"]
                attention_syn_layer3 = attention_syn[0]*syn_out["layer3"]

                # compute sorting matching losses
                loss_conv1 = sort_matching(input=attention_syn_conv1, target=attention_conv1)*importance_neuron[3].unsqueeze(-1)
                loss_layer1 = sort_matching(input=attention_syn_layer1, target=attention_layer1)*importance_neuron[2].unsqueeze(-1)
                loss_layer2 = sort_matching(input=attention_syn_layer2, target=attention_layer2)*importance_neuron[1].unsqueeze(-1)
                loss_layer3 = sort_matching(input=attention_syn_layer3, target=attention_layer3)*importance_neuron[0].unsqueeze(-1)

                loss_conv1 = loss_conv1.mean()
                loss_layer1 = loss_layer1.mean()
                loss_layer2 = loss_layer2.mean()
                loss_layer3 = loss_layer3.mean()
               
                loss_add = self.layer_weights[0]*loss_conv1 +\
                           self.layer_weights[3]*loss_layer3 +\
                           self.layer_weights[1]*loss_layer1 +\
                           self.layer_weights[2]*loss_layer2 

                # image prior losses  
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                # combining losses
                loss = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.l2_scale * loss_l2 + \
                           self.feat_dist * loss_add

                # store gradient for transparency accumulation
                grad_input = torch.autograd.grad(loss, inputs, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
                transparency_accumulator += torch.abs(grad_input)

                if iteration % print_every==0:
                    print("------------iteration {}----------".format(iteration))
                    print("total loss", loss.item())

                loss.backward(retain_graph=True)

                optimizer.step()

                # clip color outlayers
                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=False)

        # save the final image and the masked image with the transparency accumulator
        best_inputs = inputs.data.clone()
        best_inputs = denormalize(best_inputs)
        self.save_images(best_inputs)

        if self.exp_name is None:
            name = f"c1_{self.layer_weights[0]}_l1_{self.layer_weights[1]}_l2_{self.layer_weights[2]}_l3_{self.layer_weights[3]}_l4_{self.layer_weights[4]}_masked.png"
            place_to_store = os.path.join(self.folder_name, name)
        else:
            place_to_store = os.path.join(self.folder_name, self.exp_name + "_masked.png")

        save_maco(inputs.data.clone()[0], transparency_accumulator.data.clone()[0], percentile_image=1.0, percentile_alpha=98, filename=place_to_store)

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
                place_to_store = os.path.join(self.folder_name, self.exp_name + ".png")

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    

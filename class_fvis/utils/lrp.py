"""Class for layer-wise relevance propagation.

Layer-wise relevance propagation for VGG-like networks from PyTorch's Model Zoo.
Implementation can be adapted to work with other architectures as well by adding the corresponding operations.

    Typical usage example:

        model = torchvision.models.vgg16(pretrained=True)
        lrp_model = LRPModel(model)
        r = lrp_model.forward(x)

"""
from copy import deepcopy
from typing import Union

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet

from torchvision.models.densenet import DenseNet
from torchvision.models.densenet import _DenseLayer, _DenseBlock

from utils.utils import SkipConnectionPropType, layers_lookup
from utils.lrp_layers import RelevancePropagationBasicBlockFlowsPureSkip, RelevancePropagationConv2d, RelevancePropagationBottleneckFlowsPureSkip, RelevancePropagationDenseLayer

class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module, rel_pass_ratio: float = 0.0, skip_connection_prop="latest") -> None:
        super().__init__()
        self.model: Union[VGG, ResNet, DenseNet] = model
        self.rel_pass_ratio = rel_pass_ratio
        self.skip_connection_prop = skip_connection_prop

        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network (architecture must be based on VGG...)
        self.layers = self._get_layer_operations()

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """

        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup(self.skip_connection_prop)

        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer, top_k=self.rel_pass_ratio)
            except KeyError:
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)

        return layers

    def _get_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        layers = torch.nn.ModuleList()

        all_children = list(self.model.children())

        if isinstance(all_children[0], nn.Sequential):

            all_children = list(self.model.features.children()) + list(torch.nn.modules.activation.ReLU(inplace=True).modules()) +\
                 list(torch.nn.modules.pooling.AdaptiveAvgPool2d((1,1)).modules()) + list(self.model.classifier.modules())

        # Parse VGG, OneWayResNet
        for layer in all_children:
            is_tower = (isinstance(layer, nn.Sequential) and (isinstance(layer[0], BasicBlock) or isinstance(layer[0], Bottleneck))) \
                        or isinstance(layer, _DenseBlock) 
            is_linear = isinstance(layer, nn.Linear)
            if is_tower:
                for sub_layer in layer.children():
                    # assert isinstance(sub_layer, BasicBlock) or isinstance(sub_layer, Bottleneck) or isinstance(sub_layer, _DenseLayer)
                    layers.append(sub_layer)
            elif is_linear:
                layers.append(torch.nn.Flatten(start_dim=1))
                layers.append(layer)
            else:
                layers.append(layer)

        return layers

    def forward(self, x: torch.tensor, topk=-1, channel=None, return_nodes=None) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).

        """

        if return_nodes is not None: 

            fdim = return_nodes["fdim"]
            res = return_nodes["res"]
            relevance_dict = []

        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                new_x = layer.forward(x)
                if isinstance(layer, _DenseLayer):
                    new_x = torch.cat([x, new_x], 1)
                activations.append(new_x)
                x = new_x 

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        
        if channel is None:

            relevance = torch.softmax(activations.pop(0), dim=1)  # Unsupervised

            if topk != -1:
                relevance_zero = torch.zeros_like(relevance)
                top_k_indices = torch.topk(relevance, topk).indices
                for index in top_k_indices:
                    # relevance_zero[..., index] = 1.0
                    relevance_zero[..., index] = relevance[..., index]
                relevance = relevance_zero
        else:

            relevance = activations.pop(0)
            relevance_zero = torch.zeros_like(relevance)
            relevance_zero[:, channel] = relevance[:, channel]
            relevance = relevance_zero

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            a = activations.pop(0)
            try:
                relevance = layer.forward(a, relevance)
            except RuntimeError:
                print(f"RuntimeError at layer {i}.\n"
                      f"Layer: {layer.__class__.__name__}\n"
                      f"Relevance shape: {relevance.shape}\n"
                      f"Activation shape: {activations[0].shape}\n")
                exit(1)

            if len(relevance.size())>2:

                if return_nodes is not None: 
                    save_fdim = fdim[0]
                    save_res = res[0]

                    if (save_fdim == relevance.shape[1]) and (save_res == relevance.shape[2]) and self.lrp_layers[i]==self.lrp_layers[-1]:
                        relevance_dict.append(relevance)
                        fdim.pop(0)
                        res.pop(0)
                        if len(fdim)==0:
                            return relevance_dict

                    elif (save_fdim == relevance.shape[1]) and (save_res == relevance.shape[2]) and \
                    (isinstance(self.lrp_layers[i+1], RelevancePropagationBasicBlockFlowsPureSkip) or isinstance(self.lrp_layers[i+1], RelevancePropagationBottleneckFlowsPureSkip) or \
                     isinstance(self.lrp_layers[i+1], RelevancePropagationDenseLayer) or isinstance(self.lrp_layers[i+1], RelevancePropagationConv2d)):
                        
                        relevance_dict.append(relevance)
                        fdim.pop(0)
                        res.pop(0)
                        if len(fdim)==0:
                            return relevance_dict

        return relevance


# legacy code
class LRPModules(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(
        self, layers: nn.ModuleList, out_relevance: torch.Tensor, top_k: float = 0.0
    ) -> None:
        super().__init__()
        self.top_k = top_k

        # Parse network
        self.layers = layers
        self.out_relevance = out_relevance

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()

        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer, top_k=self.top_k)
            except KeyError:
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)

        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).

        """
        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised
        if self.out_relevance is not None:
            relevance = self.out_relevance.to(relevance.device)

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance)

        return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()


def basic_lrp(
    model, image, rel_pass_ratio=1.0, topk=1, skip_connection_prop: SkipConnectionPropType = "latest"
):
    lrp_model = LRPModel(model, rel_pass_ratio=rel_pass_ratio, skip_connection_prop=skip_connection_prop)
    R = lrp_model.forward(image, topk)
    return R


# Legacy code -----------------------
def resnet_lrp(model, image, topk=0.2):
    output = model(image)
    score, class_index = torch.max(output, 1)
    R = torch.zeros_like(output)
    R[0, class_index] = score

    post_modules = divide_module_by_name(model, "avgpool")
    new_post = post_modules[:-1]
    new_post.append(torch.nn.Flatten(start_dim=1))
    new_post.append(post_modules[-1])
    post_modules = new_post

    post_lrp = LRPModules(post_modules, R, top_k=topk)
    R = post_lrp.forward(post_modules[0].activations)

    R = resnet_layer_lrp(model.layer4, R, top_k=topk)
    R = resnet_layer_lrp(model.layer3, R, top_k=topk)
    R = resnet_layer_lrp(model.layer2, R, top_k=topk)
    R = resnet_layer_lrp(model.layer1, R, top_k=topk)

    pre_modules = divide_module_by_name(model, "layer1", before_module=True)
    pre_lrp = LRPModules(pre_modules, R, top_k=topk)
    R = pre_lrp.forward(image)

    return R


def abn_lrp(model, image, topk=0.2):
    output = model(image)
    score, class_index = torch.max(output, 1)
    R = torch.zeros_like(output)
    R[0, class_index] = score

    #########################
    ### Perception Branch ###
    #########################
    post_modules = nn.ModuleList(
        [
            model.perception_branch[2],
            model.perception_branch[3],
            model.perception_branch[4],
        ]
    )
    new_post = post_modules[:-1]
    new_post.append(torch.nn.Flatten(start_dim=1))
    new_post.append(post_modules[-1])
    post_modules = new_post

    post_lrp = LRPModules(post_modules, R, top_k=topk)
    R_pb = post_lrp.forward(post_modules[0].activations)

    for sequential_blocks in model.perception_branch[:2][::-1]:
        R_pb = resnet_layer_lrp(sequential_blocks, R_pb, topk)

    #########################
    ### Attention Branch  ###
    #########################
    # h -> layer1, bn1, conv1, relu, conv4, avgpool, flatten
    ab_modules = nn.ModuleList(
        [
            model.attention_branch.bn1,
            model.attention_branch.conv1,
            model.attention_branch.relu,
            model.attention_branch.conv4,
            model.attention_branch.avgpool,
            model.attention_branch.flatten,
        ]
    )
    ab_lrp = LRPModules(ab_modules, R, top_k=topk)
    R_ab = ab_lrp.forward(model.attention_branch.bn1_activation)
    R_ab = resnet_layer_lrp(model.attention_branch.layer1, R_ab, topk)

    #########################
    ### Feature Extractor ###
    #########################
    R_fe_out = R_pb + R_ab
    R = resnet_layer_lrp(model.feature_extractor[-1], R_fe_out, topk)
    R = resnet_layer_lrp(model.feature_extractor[-2], R, topk)

    pre_modules = nn.ModuleList(
        [
            model.feature_extractor[0],
            model.feature_extractor[1],
            model.feature_extractor[2],
            model.feature_extractor[3],
        ]
    )
    pre_lrp = LRPModules(pre_modules, R, top_k=topk)
    R = pre_lrp.forward(image)

    return R


def resnet_layer_lrp(
    layer: nn.Sequential, out_relevance: torch.Tensor, top_k: float = 0.0
):
    for res_block in layer[::-1]:
        inputs = res_block.activations

        identify = out_relevance
        if res_block.downsample is not None:
            downsample = nn.ModuleList(
                [res_block.downsample[0], res_block.downsample[1]]
            )
            skip_lrp = LRPModules(downsample, identify, top_k=top_k)
            skip_relevance = skip_lrp.forward(inputs)
        else:
            skip_relevance = identify

        main_modules = nn.ModuleList()
        for name, module in res_block._modules.items():
            if name == "downsample":
                continue
            main_modules.append(module)
        main_lrp = LRPModules(main_modules, identify, top_k=top_k)
        main_relevance = main_lrp.forward(inputs)

        gamma = 0.5
        out_relevance = gamma * main_relevance + (1 - gamma) * skip_relevance
    return out_relevance


def divide_module_by_name(model, module_name: str, before_module: bool = False):
    use_module = before_module
    modules = nn.ModuleList()
    for name, module in model._modules.items():
        if name == module_name:
            use_module = not use_module
        if not use_module:
            continue
        modules.append(module)

    return modules

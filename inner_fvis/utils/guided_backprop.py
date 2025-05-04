import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Tuple, Union
from torch import device, Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from inspect import signature


def _register_backward_hook(
    module: Module, hook: Callable, attr_obj: Any
) -> List[torch.utils.hooks.RemovableHandle]:
    grad_out: Dict[device, Tensor] = {}

    def forward_hook(
        module: Module,
        inp: Union[Tensor, Tuple[Tensor, ...]],
        out: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        nonlocal grad_out
        grad_out = {}

        def output_tensor_hook(output_grad: Tensor) -> None:
            grad_out[output_grad.device] = output_grad

        if isinstance(out, tuple):
            assert (
                len(out) == 1
            ), "Backward hooks not supported for module with >1 output"
            out[0].register_hook(output_tensor_hook)
        else:
            out.register_hook(output_tensor_hook)

    def pre_hook(module, inp):
        def input_tensor_hook(input_grad: Tensor):
            if len(grad_out) == 0:
                return
            hook_out = hook(module, input_grad, grad_out[input_grad.device])

            if hook_out is not None:
                return hook_out[0] if isinstance(hook_out, tuple) else hook_out

        if isinstance(inp, tuple):
            assert (
                len(inp) == 1
            ), "Backward hooks not supported for module with >1 input"
            inp[0].register_hook(input_tensor_hook)
            return inp[0].clone()
        else:
            inp.register_hook(input_tensor_hook)
            return inp.clone()

    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(forward_hook),
    ]

class LayerGuidedBackprop():
    def __init__(self, model: Module, use_relu_grad_output: bool = False) -> None:
        r"""
        Args:

            model (nn.Module): The reference to PyTorch model instance.
        """
        self.model = model
        self.backward_hooks: List[RemovableHandle] = []
        self.use_relu_grad_output = use_relu_grad_output
        assert isinstance(self.model, torch.nn.Module), (
            "Given model must be an instance of torch.nn.Module to properly hook"
            " ReLU layers."
        )

    def attribute(
        self,
        inputs,
        target_layers=None,
        target=None,
    ):
        # Register hooks for all target layers
        if target_layers is not None:
            all_activations = {layer: [] for layer in target_layers}
            # Hook for activations
            def forward_hook(module, input, output, layer_name):
                all_activations[layer_name].append(output)

            hooks = []
            for layer_name, layer in target_layers.items():
                hooks.append(layer.register_forward_hook(lambda module, input, output, ln=layer_name: forward_hook(module, input, output, ln)))

        self.model.apply(self._register_hooks)
            
        outputs = self.model(inputs)[:, target]
        
        if outputs.dim() > 2:        
            outputs = outputs.mean((1,2))
        
        attributions = []

        if target_layers is None:
            gradients = torch.autograd.grad(torch.unbind(outputs), inputs, retain_graph=True)[0]
            gradients = torch.relu(gradients)
            attributions.append(gradients)

        else:           
            for layer_name, layer in target_layers.items(): 
            
                layer_eval = all_activations[layer_name][0]              
                gradients = torch.autograd.grad(torch.unbind(outputs), layer_eval, retain_graph=True)[0]
                gradients = torch.relu(gradients)
                attributions.append(gradients)

            for hook in hooks:
                hook.remove()
            
        self._remove_hooks()
        
        return attributions

    def _register_hooks(self, module: Module):
        if isinstance(module, torch.nn.ReLU):
            hooks = _register_backward_hook(module, self._backward_hook, self)
            self.backward_hooks.extend(hooks)

    def _backward_hook(
        self,
        module: Module,
        grad_input: Union[Tensor, Tuple[Tensor, ...]],
        grad_output: Union[Tensor, Tuple[Tensor, ...]],
    ):
        to_override_grads = grad_output if self.use_relu_grad_output else grad_input
        if isinstance(to_override_grads, tuple):
            return tuple(
                F.relu(to_override_grad) for to_override_grad in to_override_grads
            )
        else:
            return F.relu(to_override_grads)

    def _remove_hooks(self):
        for hook in self.backward_hooks:
            hook.remove()
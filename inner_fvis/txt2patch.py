import torch
import torchvision.models as models
import os
import torchvision.utils as vutils
import argparse
from tqdm import tqdm

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument('--target_layer', type=str, default='layer4')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--topk_patches', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='resnet50/neuron_layer4')
parser.add_argument('--channel', type=int, default=1935)

args = parser.parse_args()
print(args)

def get_activation_map(model, layer_name):
    activations = {}

    def hook_fn(module, input, output):
        activations['feat'] = output

    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(hook_fn)
    return activations

def generate_model(arch='resnet50'):
    net = models.__dict__[arch](pretrained=True).cuda().eval()
    return net

def extract_top_patches(model, layer_name, patch_size, channel, topk_patches):
    from PIL import Image
    from torchvision import transforms
    import torch.nn.functional as F
    import numpy as np
    
    save_dir = f'{args.save_dir}/{channel}/'
    
    with open(save_dir+'files_all.txt', 'r') as f:
        img_list = [line.strip() for line in f]

    def patchify(inputs, patch_size=64):
        stride = int(patch_size * 0.50)
        patches = F.unfold(inputs, kernel_size=patch_size, stride=stride)
        patches = patches.transpose(1, 2).contiguous().view(-1, inputs.shape[1], patch_size, patch_size)
        return patches, stride

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    act_map = get_activation_map(model, layer_name)

    patches, patch_scores = [], []

    for path in tqdm(img_list):
        img = transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            img.requires_grad = False
            patch_batch, _ = patchify(img, patch_size=patch_size)
            _ = model(patch_batch)
            ch_activations = act_map['feat'].mean((2, 3))
            ch_activations = ch_activations[:, channel].detach().cpu()

        best_idx = torch.argmax(ch_activations).item()
        best_score = ch_activations[best_idx].item()

        patches.append(patch_batch[best_idx].cpu())
        patch_scores.append(best_score)

        del img, patch_batch
        torch.cuda.empty_cache()
        
    patch_scores = np.array(patch_scores)
    topk_idx = np.argsort(patch_scores)[::-1][:topk_patches]
    os.makedirs(save_dir, exist_ok=True)

    vutils.save_image(torch.stack([patches[i] for i in topk_idx[:16]]),
                        os.path.join(save_dir, 'patches.png'),
                        normalize=True, scale_each=True, nrow=4)
    
    # Save top-k filenames
    with open(os.path.join(save_dir, 'topk_files.txt'), 'w') as f:
        for i in topk_idx:
            f.write(f"{img_list[i]}\n")

if __name__ == '__main__':
    model = generate_model(args.arch)

    extract_top_patches(
        model=model,
        layer_name=args.target_layer,
        patch_size=args.patch_size,
        channel=args.channel,
        topk_patches=args.topk_patches
    )

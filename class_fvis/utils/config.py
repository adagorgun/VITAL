
CONFIGS = {
    "resnet50": {
        "arch_name": "resnet50",
        "layer_weights": [1, 1, 1, 1, 1],
        "tv_l2": 0.0,
        "l2": 0.0,
    },
    "densenet121": {
        "arch_name": "densenet121",
        "layer_weights": [1, 1, 1, 1, 100],
        "tv_l2": 0.0,
        "l2": 0.0,
    },
    "vit_l_16": {
        "arch_name": "vit_l_16",
        "layer_weights": [1, 1, 1, 1, 1],
        "tv_l2": 0.003,
        "l2": 0.003,
    },
    "vit_l_32": {
        "arch_name": "vit_l_32",
        "layer_weights": [1, 1, 1, 1, 0.1],
        "tv_l2": 0.0003,
        "l2": 0.0003,
    },
    "convnext_base": {
        "arch_name": "convnext_base",
        "layer_weights": [10, 1, 1, 1, 1],
        "tv_l2": 0.0,
        "l2": 0.0,
    }
}

def get_config(args):
    arch_name = args.arch_name

    defaults = CONFIGS[arch_name]
    args.layer_weights = getattr(args, 'layer_weights', defaults['layer_weights'])
    args.tv_l2 = getattr(args, 'tv_l2', defaults['tv_l2'])
    args.l2 = getattr(args, 'l2', defaults['l2'])

    # universal settings
    args.folder_name = getattr(
        args,
        'folder_name',
        f"class_neurons/{arch_name}/VITAL/rand{args.num_real_img}/cls{args.target}/t{args.run_id}"
    )

    return args
import torch
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# import utils.comm as comm
from models.deit import DeiT_S_16
from models.models import load_model


def get_class_model(num_classes, device, checkpoint):
    net = DeiT_S_16(num_classes=num_classes)

    print("Loading Checkpoint")
    checkpoint_state_dict = torch.load(checkpoint, map_location="cpu")

    if 'model' in checkpoint_state_dict:
        checkpoint_state_dict = checkpoint_state_dict['model']
        new_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            new_key = "model." + key
            if 'head' in new_key:
                new_state_dict[new_key] = value
                new_key = key.replace('model.head.', 'fc.', 1)
            new_state_dict[new_key] = value
    else:
        new_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if 'backbone' in key:
                new_key = key.replace('backbone.', '', 1)  # Remove 'model.' prefix
            elif 'head' in key:
                new_state_dict[key] = value
                new_key = key.replace('model.head.', 'fc.', 1) 
            else:
                new_key = key
            new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict, strict=False)

    # Check if the keys of the checkpoint match the keys of the model's state dictionary
    model_state_dict = net.state_dict()
    checkpoint_keys = set(new_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    # Check for keys present in the checkpoint but missing in the model
    missing_keys = model_keys - checkpoint_keys
    if missing_keys:
        print("Keys missing in the model's state dictionary:", missing_keys)

    # Check for keys present in the model but missing in the checkpoint
    unexpected_keys = checkpoint_keys - model_keys
    if unexpected_keys:
        print("Unexpected keys in the checkpoint:", unexpected_keys)

    # Check if the sizes of the tensors match between the model and the checkpoint
    for key in model_keys.intersection(checkpoint_keys):
        if model_state_dict[key].shape != new_state_dict[key].shape:
            print(f"Shape mismatch for key '{key}': checkpoint shape {new_state_dict[key].shape}, model shape {model_state_dict[key].shape}")
    print('Classification Model Loading DeiT Completed!')

    net.to(device)
    return net

def get_obj_detect_model(checkpoint, device, args):
    model_path = "/home/shpark/Colood/config/Config.cfg"
    model = load_model(model_path, checkpoint, args=args)
    model.to(device)
    return model

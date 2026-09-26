import torch


def update_dict_to_target(target_dict, source_dict, factor=1.0):
    for k, v in source_dict.items():
        if k not in target_dict:
            target_dict[k] = []
        if torch.is_tensor(v):
            if v.numel() == 1:
                target_dict[k].append(v.detach().item() * factor)
            else:
                target_dict[k].append(v.detach() * factor)
        else:
            target_dict[k].append(v * factor)
    return target_dict


def update_dict_to_output(output_dict, source_dict, f='f', decimal=4):
    for k, v in source_dict.items():
        output_dict[k] = f"{v:.{decimal}{f}}"
    return output_dict

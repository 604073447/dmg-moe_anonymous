



def unpack_batch(batch, device):
    inputs = batch["image"].to(device, non_blocking=True)
    targets = {
        'age': batch["label"].to(device, non_blocking=True),
        'weight': batch["weight"].to(device, non_blocking=True),
    }
    return inputs, targets

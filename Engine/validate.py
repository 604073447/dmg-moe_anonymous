import pandas as pd
import torch
import torch.distributed as dist
from tabulate import tabulate
from tqdm import tqdm

from Engine.metrics import calculate_metrics_reg, calculate_metrics_shot
from Engine.utils import unpack_batch
from Utils.distributed import is_master, gather_variable_length_tensor
from Utils.predefine import SHOT_LIST, met_map, metric_of_interest
from Utils.for_concise import value_fmt


# Loads training labels used for shot-group metrics.
def _get_train_labels(config):
    cache = getattr(_get_train_labels, "_cache", {})
    key = (config.data.train_csv, config.data.label_col)
    if key not in cache:
        cache[key] = pd.read_csv(config.data.train_csv)[config.data.label_col].to_numpy(dtype=int)
        setattr(_get_train_labels, "_cache", cache)
    return cache[key]


# Builds the validation or test metric table.
def _build_metrics_table(config, mode, preds, targs, avg_loss):
    reg = calculate_metrics_reg(preds, targs)
    reg["loss"] = avg_loss
    shot = calculate_metrics_shot(preds, targs, _get_train_labels(config), **config.data.shot_thr_dict)
    table = {row: {} for row in SHOT_LIST}
    table["All"] = {met_map[k]: v for k, v in reg.items()}
    table["Many"] = {met_map[k]: v for k, v in shot["many"].items()}
    table["Med."] = {met_map[k]: v for k, v in shot["median"].items()}
    table["Few"] = {met_map[k]: v for k, v in shot["low"].items()}
    table["Macro"] = {}
    for key in table["Many"].keys():
        if key == "GM":
            table["Macro"][key] = (table["Many"][key] * table["Med."][key] * table["Few"][key]) ** (1.0 / 3.0)
        else:
            table["Macro"][key] = (table["Many"][key] + table["Med."][key] + table["Few"][key]) / 3.0
    columns = sorted({col for row in SHOT_LIST for col in table[row].keys()})
    df = pd.DataFrame(index=SHOT_LIST, columns=columns)
    for row in SHOT_LIST:
        for col in columns:
            df.loc[row, col] = table[row].get(col, "-")
    df.index.name = mode
    df_print = df.map(value_fmt)
    selected = [col for col in metric_of_interest if col in df_print.columns]
    if selected:
        df_print = df_print[selected]
    table_str = tabulate(df_print, headers="keys", tablefmt="fancy_grid", numalign="right", stralign="left", floatfmt=".4f", showindex=True)
    return table, table_str, df


# Evaluates a model on one split and returns metrics.
@torch.no_grad()
def validate(config, logger, device, model, loader, criterion, epoch, num_epochs, desc="", mode="Valid", show_pbar=False, save_name=None):
    model.eval()
    all_preds = []
    all_targs = []
    total_loss = 0.0
    num_samples = 0
    iterator = tqdm(loader, desc=desc) if show_pbar and is_master() else loader
    for batch in iterator:
        inputs, targets = unpack_batch(batch, device)
        outputs = model(inputs, epoch=epoch)
        loss, _, _ = criterion(outputs, targets, return_dict=True) if criterion is not None else (torch.tensor(0.0, device=device), None, None)
        batch_size = int(inputs.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        num_samples += batch_size
        all_preds.append(outputs["age"].detach().view(-1))
        all_targs.append(targets["age"].detach().view(-1))
    preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty((0,), device=device)
    targs = torch.cat(all_targs, dim=0) if all_targs else torch.empty((0,), device=device)
    preds = gather_variable_length_tensor(preds, concat=True)
    targs = gather_variable_length_tensor(targs, concat=True)
    if dist.is_initialized():
        sync = torch.tensor([total_loss, float(num_samples)], device=device)
        dist.all_reduce(sync, op=dist.ReduceOp.SUM)
        total_loss = sync[0].item()
        num_samples = int(sync[1].item())
    broadcast_data = [None, None]
    if is_master():
        table, table_str, df = _build_metrics_table(config, mode, preds, targs, total_loss / max(num_samples, 1))
        if logger is not None:
            logger.info("\n" + table_str)
        if save_name is not None:
            df.to_csv(f"{config.path.exp_dir}/{save_name}", index=True)
        broadcast_data[0] = table
        broadcast_data[1] = table_str
    if dist.is_initialized():
        dist.broadcast_object_list(broadcast_data, src=0)
    return broadcast_data[0], broadcast_data[1]

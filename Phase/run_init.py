import copy
import os

import torch

from Engine.loops import train_one_epoch
from Engine.validate import validate
from Utils.checkpoint import save_checkpoint
from Utils.distributed import wrap_model, unwrap_model, log_info, is_master
from Utils.predefine import lower_better_metrics
from App.bootstrap import build_optimizer, build_scheduler, build_criterion, build_earlystop, build_normscale, build_ema_model


# Initializes training state for the main phase.
def init_define(config, raw_model):
    phase = 'Init'
    cfg = config.train

    best_record = {
        'state_dict': copy.deepcopy(raw_model.state_dict()),
        'stats': None,
        'table': None,
        'score': float('inf') if cfg.met_name in lower_better_metrics else -float('inf'),
        'epoch': -1,
        'mode': 'min' if cfg.met_name in lower_better_metrics else 'max',
        'test_table': None,
        'test_score': None,
    }
    return phase, cfg, best_record


# Creates DDP model, optimizer, loss, scheduler, early stopping, and EMA.
def setup_module(cfg, phase, raw_model, logger, best_record, ddp_info):
    criterion = build_criterion(cfg).cuda()
    optimizer = build_optimizer(cfg, phase, raw_model, logger=logger)
    scheduler = build_scheduler(cfg, optimizer, num_epochs=cfg.num_epochs)
    normscale = build_normscale(cfg)
    earlystop = build_earlystop(cfg, best_record)
    ema_model = build_ema_model(cfg, raw_model, device=torch.device(f"cuda:{ddp_info['local_rank']}"))

    ddp_model = wrap_model(raw_model, ddp_info, cfg.find_unused_parameters)
    return ddp_model, optimizer, criterion, normscale, scheduler, earlystop, ema_model


def _maybe_step_scheduler(scheduler, curr_perf: float):
    if scheduler is None:
        return
    scheduler.step(curr_perf)


# Runs one train/validation/test epoch and updates the best record.
def do_experiment(config, cfg, device, ddp_model, optimizer, criterion, normscale, scheduler, earlystop,
                  train_sampler, train_loader, valid_loader, test_loader, ema_model,
                  logger, phase, epoch, best_record,
                  global_step,):
    epoch_desc = f"Init Epoch {epoch + 1}/{cfg.num_epochs}"

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_stats, global_step = train_one_epoch(
        logger, device,
        ddp_model, optimizer, criterion, normscale,
        train_loader, ema_model,
        cfg.autocast_kwargs, cfg.grad_scaler_enabled, cfg.gradnorm_kwargs,
        epoch, global_step,
        epoch_desc,
    )

    eval_model = ema_model.ema if (ema_model is not None and cfg.eval_with_ema) else ddp_model

    valid_stats, valid_table = validate(
        config=config,
        logger=logger,
        device=device,
        model=eval_model,
        loader=valid_loader,
        criterion=criterion,
        epoch=epoch,
        num_epochs=cfg.num_epochs,
        desc=epoch_desc,
        mode='Valid',
    )
    curr_perf = valid_stats[cfg.maj_shot][cfg.met_name]

    test_table, test_perf = None, None
    if test_loader is not None:
        test_stats, test_table = validate(
            config=config,
            logger=logger,
            device=device,
            model=eval_model,
            loader=test_loader,
            criterion=criterion,
            epoch=epoch,
            num_epochs=cfg.num_epochs,
            desc=epoch_desc,
            mode='Test',
        )
        test_perf = test_stats[cfg.maj_shot][cfg.met_name]

    _maybe_step_scheduler(scheduler, curr_perf)

    should_stop = earlystop(curr_perf, epoch, test_perf)
    if earlystop.improved:
        best_test_str = f"{earlystop.best_test_score:.4f}" if earlystop.best_test_score is not None else "N/A"
        log_info(logger, f"Significant improvement detected. "
                         f"Best record: {earlystop.best_score:.4f} (Raw: {earlystop.best_raw_score:.4f}) "
                         f"(Test: {best_test_str}) at Epoch {earlystop.best_epoch}.\n")
        best_record.update({
            'state_dict': copy.deepcopy(unwrap_model(eval_model).state_dict()),
            'stats': valid_stats,
            'table': valid_table,
            'score': curr_perf,
            'epoch': epoch,
            'test_table': test_table,
            'test_score': test_perf,
        })
    else:
        best_test_str = f"{earlystop.best_test_score:.4f}" if earlystop.best_test_score is not None else "N/A"
        log_info(logger, f"No significant improvement [{earlystop.counter}/{earlystop.patience}]. "
                         f"Best record: {earlystop.best_score:.4f} (Raw: {earlystop.best_raw_score:.4f}) "
                         f"(Test: {best_test_str}) at Epoch {earlystop.best_epoch}.\n")

    return train_stats, valid_stats, valid_table, curr_perf, should_stop, global_step, eval_model


# Saves the best model and final checkpoint.
def process_result(ddp_model, best_record, exp_dir, logger, phase='Init'):
    best_model = unwrap_model(ddp_model)
    best_model.load_state_dict(best_record['state_dict'])

    if is_master():
        save_path = os.path.join(exp_dir, "best_root_init_model.pth")
        torch.save(best_model, save_path)
        logger.info(f"Best model saved to {save_path}.")
        logger.info(f"Best stats at epoch {best_record['epoch']}:")
        if best_record.get('table') is not None:
            logger.info("\n" + best_record['table'])
        if best_record.get('test_table') is not None:
            logger.info("\n" + best_record['test_table'])
    save_checkpoint(f"{exp_dir}/checkpoint_{phase}.pth", best_model, best_record, logger)
    return best_model, best_record


# Runs the full initial training phase.
def run_init(
    config, device, raw_model, ddp_info,
    train_loader, train_sampler, valid_loader, test_loader=None,
    logger=None,
):
    phase, cfg, best_record = init_define(config, raw_model)

    ddp_model, optimizer, criterion, normscale, scheduler, earlystop, ema_model\
        = setup_module(cfg, phase, raw_model, logger, best_record, ddp_info)
    eval_model = ema_model.ema if (ema_model is not None and cfg.eval_with_ema) else ddp_model

    global_step = 0
    for epoch in range(cfg.num_epochs):
        train_stats, valid_stats, valid_table, curr_perf, should_stop, global_step, eval_model = do_experiment(
            config, cfg, device, ddp_model, optimizer, criterion, normscale, scheduler, earlystop,
            train_sampler, train_loader, valid_loader, test_loader, ema_model,
            logger, phase, epoch, best_record,
            global_step,
        )

        if epoch % 10 == 0:
            save_model = unwrap_model(ddp_model)
            save_model.load_state_dict(best_record['state_dict'])
            save_path = os.path.join(config.path.exp_dir, f"root_init_model_{epoch}.pth")
            torch.save(save_model, save_path)

        if should_stop:
            log_info(logger, f"[root] Early stopping at epoch {epoch + 1}")
            break

    best_model, best_record = process_result(eval_model, best_record, config.path.exp_dir, logger)
    return best_model, best_record

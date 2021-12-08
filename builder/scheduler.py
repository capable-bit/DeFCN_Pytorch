from optimizer_scheduler.scheduler import WarmupMultiStepLR


def build_lr_scheduler(cfg, optimizer):
    scheduler = WarmupMultiStepLR(
                optimizer,
                cfg.SOLVER.LR_SCHEDULER.STEPS,
                cfg.SOLVER.LR_SCHEDULER.GAMMA,
                warmup_factor=cfg.SOLVER.LR_SCHEDULER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.LR_SCHEDULER.WARMUP_METHOD,
            )

    return scheduler

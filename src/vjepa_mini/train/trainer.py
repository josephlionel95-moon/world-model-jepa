"""Training loop for mini V-JEPA, sized for a free Colab T4.

Design choices worth noticing:
  * Mixed precision (AMP): halves memory, ~2x speed on T4 tensor cores.
  * One mask per batch, shared across clips (see data/masking.py).
  * The EMA update happens AFTER the optimizer step, on the new weights.
  * We log target_std — the collapse early-warning signal. Healthy runs sit
    well above 0; a slide toward 0 means the representation is dying.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from vjepa_mini.config import VJEPAConfig
from vjepa_mini.data.masking import VJEPAMasks
from vjepa_mini.models.vjepa import VJEPA
from vjepa_mini.train.schedules import cosine_with_warmup, linear_schedule
from vjepa_mini.utils import AverageMeter, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: VJEPA,
        cfg: VJEPAConfig,
        loader: DataLoader,
        device: torch.device,
        ckpt_dir: str = "./checkpoints",
        log_every: int = 50,
        on_log: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.loader = loader
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.log_every = log_every
        self.on_log = on_log
        self.masks = VJEPAMasks(cfg)

        # AdamW: decoupled weight decay (wd set per-step from the schedule).
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.wd_start,
            betas=(0.9, 0.999),
        )
        self.scaler = torch.amp.GradScaler(enabled=cfg.use_amp)
        self.history: List[Dict[str, float]] = []
        self.step = 0

    def _set_hparams(self) -> Dict[str, float]:
        """Apply this step's lr / wd; return them plus EMA momentum."""
        lr = cosine_with_warmup(
            self.step, self.cfg.total_steps, self.cfg.lr, self.cfg.final_lr,
            self.cfg.warmup_steps, self.cfg.schedule_scale,
        )
        wd = linear_schedule(
            self.step, self.cfg.total_steps, self.cfg.wd_start, self.cfg.wd_end,
            self.cfg.schedule_scale,
        )
        m = linear_schedule(
            self.step, self.cfg.total_steps, self.cfg.ema_start, self.cfg.ema_end,
            self.cfg.schedule_scale,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd
        return {"lr": lr, "wd": wd, "momentum": m}

    def train(self, total_steps: Optional[int] = None) -> List[Dict[str, float]]:
        total = total_steps or self.cfg.total_steps
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss", "target_std", "pred_std")}
        data_iter = iter(self.loader)

        while self.step < total:
            try:
                video, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                video, _ = next(data_iter)
            video = video.to(self.device, non_blocking=True)

            hp = self._set_hparams()
            context_idx, target_idx = self.masks()
            context_idx = context_idx.to(self.device)
            target_idx = target_idx.to(self.device)

            with torch.autocast(
                device_type=self.device.type, enabled=self.cfg.use_amp
            ):
                out = self.model(video, context_idx, target_idx)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(out["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.encoder.parameters(), self.cfg.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update AFTER the optimizer step, with the scheduled momentum.
            self.model.update_target_encoder(hp["momentum"])

            for k in meters:
                meters[k].update(out[k].item())
            self.step += 1

            if self.step % self.log_every == 0:
                record = {
                    "step": self.step,
                    **{k: meters[k].avg for k in meters},
                    **hp,
                }
                self.history.append(record)
                if self.on_log:
                    self.on_log(record)
                else:
                    print(
                        f"step {record['step']:>6d} | loss {record['loss']:.4f} "
                        f"| target_std {record['target_std']:.4f} "
                        f"| lr {record['lr']:.2e}"
                    )
                for m in meters.values():
                    m.sum, m.count = 0.0, 0

        self.save(self.ckpt_dir / "vjepa_final.pt")
        return self.history

    def save(self, path) -> None:
        save_checkpoint(
            path,
            encoder=self.model.encoder.state_dict(),
            target_encoder=self.model.target_encoder.state_dict(),
            predictor=self.model.predictor.state_dict(),
            optimizer=self.optimizer.state_dict(),
            cfg=self.cfg,
            step=self.step,
            history=self.history,
        )

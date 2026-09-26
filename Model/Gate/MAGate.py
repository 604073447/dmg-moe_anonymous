import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Block.MLP import MLP


# Aligns gate probabilities with label-aware feature mismatch.
class MAGate(nn.Module):
    def __init__(
        self,
        gid: int,
        in_channels: int,
        proj_cfg: dict,
        tau_z: int = 1.0,
        tau_y: int = 1.0,
        detach_gate_input: bool = False,
        detach_z_for_gloss: bool = True,
        sample_unique_age: bool = False,
        norm_z_logits: bool = False,
    ):
        super().__init__()
        self.tau_z = tau_z
        self.tau_y = tau_y
        self.detach_gate_input = detach_gate_input
        self.detach_z_for_gloss = detach_z_for_gloss
        self.sample_unique_age = sample_unique_age
        self.norm_z_logits = norm_z_logits
        self.gid = gid
        self.in_channels = in_channels
        self.mlp_kwargs = proj_cfg['mlp_kwargs']
        self.num_layers = proj_cfg['num_layers']
        self.hidden_dim = proj_cfg['hidden_dim'] if self.num_layers > 0 else in_channels
        self.gate_proj = self._build_gate_projection() if self.num_layers > 0 else None
        self.gate_head = nn.Linear(self.hidden_dim, 1)

    def _build_gate_projection(self):
        layers = []
        in_dim = self.in_channels
        for _ in range(self.num_layers):
            layers.append(MLP(in_dim, self.hidden_dim, **self.mlp_kwargs))
            in_dim = self.hidden_dim
        return nn.Sequential(*layers)

    def _predict_gate(self, z: torch.Tensor):
        proj = self.gate_proj(z) if self.gate_proj is not None else z
        logit = self.gate_head(proj)
        return torch.sigmoid(logit).squeeze(-1)

    # Computes the gate probability and its auxiliary target loss.
    def forward(self, z, y_gt=None, eps=1e-6, epoch=None, idx=None):
        z_for_gate = z.detach() if self.detach_gate_input else z
        gate_prob = self._predict_gate(z_for_gate)
        gate_loss = z.new_tensor(0.0)
        batch_size = z.size(0)

        if self.training and (y_gt is not None) and (batch_size > 1):
            with torch.no_grad():
                z_u = z
                y_u = y_gt
                first_idx = None

                if self.sample_unique_age:
                    uniq_y, inv = torch.unique(y_gt.view(-1), sorted=False, return_inverse=True)
                    num_unique = int(uniq_y.numel())
                    idx_all = torch.arange(batch_size, device=y_gt.device, dtype=torch.long)
                    first_idx = torch.full((num_unique,), batch_size, device=y_gt.device, dtype=torch.long)
                    first_idx.scatter_reduce_(0, inv, idx_all, reduce="amin", include_self=True)
                    if num_unique > 1:
                        z_u = z[first_idx]
                        y_u = y_gt[first_idx]

                m_cur = self._mismatch_kl(z_u, y_u)
                teacher = torch.exp(-m_cur)
                teacher = teacher.clamp(eps, 1.0 - eps).unsqueeze(1)

            if self.detach_z_for_gloss:
                student_for_loss = self._predict_gate(z.detach()).unsqueeze(1)
            else:
                student_for_loss = gate_prob.unsqueeze(1)

            if self.sample_unique_age and first_idx is not None and first_idx.numel() > 1:
                student_for_loss = student_for_loss[first_idx]
            gate_loss = F.mse_loss(student_for_loss, teacher)

        return {
            'gate_prob': gate_prob,
            'gate_loss': gate_loss,
        }

    # Measures feature-label distribution mismatch for gate supervision.
    def _mismatch_kl(self, z: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        z_norm = F.normalize(z, p=2, dim=1)
        z_logits = z_norm @ z_norm.t()
        z_logits_norm = self.row_zscore_offdiag(z_logits) / self.tau_z if self.norm_z_logits else z_logits / self.tau_z
        log_pz = self._masked_log_softmax(z_logits_norm)

        y = y_gt.view(-1, 1).float()
        dy = (y - y.t()).abs()
        y_logits = -dy / self.tau_y
        log_py = self._masked_log_softmax(y_logits)
        py = log_py.exp()

        diag = torch.eye(batch_size, device=z.device, dtype=torch.bool)
        log_py_safe = log_py.masked_fill(diag, 0.0)
        log_pz_safe = log_pz.masked_fill(diag, 0.0)
        return (py * (log_py_safe - log_pz_safe)).sum(dim=1)

    # Normalizes off-diagonal logits row by row.
    @staticmethod
    def row_zscore_offdiag(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        batch_size = logits.size(0)
        diag = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        x = logits.masked_fill(diag, 0.0)
        n = batch_size - 1
        mean = x.sum(dim=1, keepdim=True) / n
        var = ((x - mean) ** 2).masked_fill(diag, 0.0).sum(dim=1, keepdim=True) / n
        std = torch.sqrt(var + eps)
        return (logits - mean) / std

    # Computes log-softmax after masking self-pairs.
    @staticmethod
    def _masked_log_softmax(logits: torch.Tensor, mask_value=-1e9) -> torch.Tensor:
        batch_size = logits.size(0)
        mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, mask_value)
        return F.log_softmax(logits, dim=1)

import torch
from copy import deepcopy
from typing import List, Optional


class PCGrad:
    """
    PCGrad: Gradient Surgery for Multi-Task Learning.
    Projects conflicting task gradients onto each other's normal planes
    to reduce destructive interference on shared parameters.

    Reference:
        Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
        https://arxiv.org/abs/2001.06782
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optim = optimizer

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optim

    def zero_grad(self) -> None:
        self._optim.zero_grad(set_to_none=True)

    def step(self) -> None:
        self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    def pc_backward(self, losses: List[torch.Tensor]) -> None:
        """
        Compute PCGrad gradients from a list of per-task scalar losses.
        Replaces the standard loss.backward() call.

        Args:
            losses: list of scalar tensors, one per task.
        """
        # Collect per-task gradients
        task_grads = self._collect_task_grads(losses)

        # Project conflicting gradients
        projected = self._project_conflicting(task_grads)

        # Sum projected gradients and assign to parameters
        self._apply_grad(projected)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _collect_task_grads(
        self, losses: List[torch.Tensor]
    ) -> List[List[Optional[torch.Tensor]]]:
        """Backprop each task loss separately and store gradients."""
        task_grads = []
        for loss in losses:
            self._optim.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grads = []
            for group in self._optim.param_groups:
                for p in group['params']:
                    grads.append(p.grad.clone() if p.grad is not None else None)
            task_grads.append(grads)
        return task_grads

    def _project_conflicting(
        self, task_grads: List[List[Optional[torch.Tensor]]]
    ) -> List[List[Optional[torch.Tensor]]]:
        """
        For each task i, project out components of other tasks j
        whose gradients conflict (negative cosine similarity) with task i.
        """
        proj_grads = deepcopy(task_grads)

        for i in range(len(proj_grads)):
            for j in range(len(task_grads)):
                if i == j:
                    continue

                # Flatten non-None gradients for cosine computation
                g_i_parts = [g for g in proj_grads[i] if g is not None]
                g_j_parts = [g for g in task_grads[j] if g is not None]
                if not g_i_parts or not g_j_parts:
                    continue

                g_i_flat = torch.cat([g.flatten() for g in g_i_parts])
                g_j_flat = torch.cat([g.flatten() for g in g_j_parts])

                dot = torch.dot(g_i_flat, g_j_flat)

                # Only project if gradients conflict
                if dot < 0:
                    scale = dot / (g_j_flat.norm() ** 2 + 1e-12)
                    param_idx = 0
                    for k, g_i_k in enumerate(proj_grads[i]):
                        if g_i_k is None:
                            continue
                        g_j_k = task_grads[j][k]
                        if g_j_k is not None:
                            proj_grads[i][k] = g_i_k - scale * g_j_k
                        param_idx += 1

        return proj_grads

    def _apply_grad(
        self, proj_grads: List[List[Optional[torch.Tensor]]]
    ) -> None:
        """Sum projected gradients across tasks and assign to parameters."""
        n_params = len(proj_grads[0])
        summed: List[Optional[torch.Tensor]] = [None] * n_params

        for grads in proj_grads:
            for k, g in enumerate(grads):
                if g is None:
                    continue
                summed[k] = g if summed[k] is None else summed[k] + g

        param_idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if summed[param_idx] is not None:
                    p.grad = summed[param_idx]
                param_idx += 1

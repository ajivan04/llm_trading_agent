"""Differentiable categorical action statistics without prompt-vocabulary tensors."""
from __future__ import annotations

from typing import Any


def categorical_action_statistics(
    torch: Any, logits: Any, targets: Any, *, reference_logits: Any | None = None,
    chunk_size: int = 32,
) -> tuple[Any, Any, Any]:
    """Return sampled log probabilities, entropy and exact KL(pi || reference).

    Only action positions are accepted. Vocabulary reductions use float32 in
    bounded token chunks; the caller never needs prompt-length float32 logits.
    Reference probabilities are constant, while pi participates in the complete
    categorical KL gradient (unlike a signed sampled log-probability difference).
    """
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("statistics chunk size must be positive")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] <= 0:
        raise ValueError("action logits must contain one nonempty token batch")
    if targets.shape != logits.shape[:2]:
        raise ValueError("action targets must align with projected logits")
    if reference_logits is not None and reference_logits.shape != logits.shape:
        raise ValueError("reference action logits must align with policy logits")
    logprobs, entropies, divergences = [], [], []
    for start in range(0, logits.shape[1], chunk_size):
        stop = start + chunk_size
        log_distribution = torch.log_softmax(logits[:, start:stop, :].float(), dim=-1)
        distribution = log_distribution.exp()
        logprobs.append(log_distribution.gather(
            -1, targets[:, start:stop].unsqueeze(-1)).squeeze(-1))
        entropies.append(-(distribution * log_distribution).sum(dim=-1))
        if reference_logits is None:
            divergences.append(torch.zeros_like(entropies[-1]))
        else:
            with torch.no_grad():
                reference = torch.log_softmax(
                    reference_logits[:, start:stop, :].float(), dim=-1)
            divergences.append((distribution * (log_distribution - reference)).sum(dim=-1))
    return tuple(torch.cat(parts, dim=1) for parts in (logprobs, entropies, divergences))

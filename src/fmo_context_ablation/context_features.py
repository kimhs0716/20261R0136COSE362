from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LABELS = ("eta", "tau_transfer", "ipr", "purity", "c_l1")
DEFAULT_POP_INDICES = (1, 3, 4)  # 기존 reconstructed: t = 1, 5, 10 ps 인덱스


@dataclass(frozen=True)
class ContextConfig:
    name: str
    include_labels: bool = True
    include_eigs: bool = False
    include_pop_t: bool = False
    include_dynamics_summary: bool = False
    pop_indices: tuple[int, ...] = DEFAULT_POP_INDICES


CONTEXT_CONFIGS: dict[str, ContextConfig] = {
    "c5": ContextConfig("c5"),
    "c12": ContextConfig("c12", include_eigs=True),
    "c26": ContextConfig("c26", include_pop_t=True),
    "c33": ContextConfig("c33", include_eigs=True, include_pop_t=True),
    "c18": ContextConfig("c18", include_dynamics_summary=True),
    "c25": ContextConfig("c25", include_eigs=True, include_dynamics_summary=True),
}


def list_contexts() -> list[str]:
    return sorted(CONTEXT_CONFIGS)


def build_context(d: np.lib.npyio.NpzFile | dict[str, np.ndarray], name: str) -> tuple[np.ndarray, list[str]]:
    """NPZ 내용에서 선택한 condition matrix를 만든다."""
    if name not in CONTEXT_CONFIGS:
        raise KeyError(f"Unknown context '{name}'. Available: {', '.join(list_contexts())}")
    cfg = CONTEXT_CONFIGS[name]
    blocks: list[np.ndarray] = []
    names: list[str] = []

    if cfg.include_labels:
        block, block_names = _label_block(d)
        blocks.append(block)
        names.extend(block_names)
    if cfg.include_eigs:
        block, block_names = _eigs_block(d)
        blocks.append(block)
        names.extend(block_names)
    if cfg.include_pop_t:
        block, block_names = _pop_block(d, cfg.pop_indices)
        blocks.append(block)
        names.extend(block_names)
    if cfg.include_dynamics_summary:
        block, block_names = _dynamics_summary_block(d)
        blocks.append(block)
        names.extend(block_names)

    if not blocks:
        raise ValueError(f"Context '{name}' selected no features.")
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def _require_keys(d: np.lib.npyio.NpzFile | dict[str, np.ndarray], keys: tuple[str, ...] | list[str]) -> None:
    files = set(d.files if hasattr(d, "files") else d.keys())
    missing = sorted(set(keys).difference(files))
    if missing:
        raise KeyError(f"Missing keys for selected context: {missing}")


def _label_block(d: np.lib.npyio.NpzFile | dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    _require_keys(d, LABELS)
    return np.stack([d[k] for k in LABELS], axis=1).astype(np.float32), list(LABELS)


def _eigs_block(d: np.lib.npyio.NpzFile | dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    _require_keys(d, ("eigs",))
    eigs = np.sort(np.asarray(d["eigs"], dtype=np.float32), axis=1)
    return eigs, [f"eig_sorted_{i}" for i in range(eigs.shape[1])]


def _pop_block(
    d: np.lib.npyio.NpzFile | dict[str, np.ndarray],
    pop_indices: tuple[int, ...] = DEFAULT_POP_INDICES,
) -> tuple[np.ndarray, list[str]]:
    _require_keys(d, ("pop_t",))
    pop = np.asarray(d["pop_t"], dtype=np.float32)[:, list(pop_indices), :7]
    names = [f"pop_tidx{ti}_site{s + 1}" for ti in pop_indices for s in range(7)]
    return pop.reshape(len(pop), -1), names


def _dynamics_summary_block(d: np.lib.npyio.NpzFile | dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Trajectory 전체를 쓰지 않고 요약값만 뽑는 후보 condition.

    c_l1_t, purity_t, ipr_t는 label과 가깝기 때문에 이 후보를 쓸 때는
    circularity 가능성을 별도로 보고서에 명시해야 한다.
    """
    _require_keys(d, ("pop_t", "times", "cl1_t", "purity_t", "ipr_t"))
    pop_t = np.asarray(d["pop_t"], dtype=np.float32)
    times = np.asarray(d["times"], dtype=np.float32)
    c_l1_t = np.asarray(d["cl1_t"], dtype=np.float32)
    purity_t = np.asarray(d["purity_t"], dtype=np.float32)
    ipr_t = np.asarray(d["ipr_t"], dtype=np.float32)

    source_final = pop_t[:, -1, 0:1]
    sink_final = pop_t[:, -1, 2:3]
    rest_idx = [1, 3, 4, 5, 6]
    rest_final_sorted = np.sort(pop_t[:, -1, rest_idx], axis=1)

    sink_pop = pop_t[:, :, 2]
    sink_argmax = np.argmax(sink_pop, axis=1)
    sink_max = sink_pop[np.arange(len(sink_pop)), sink_argmax][:, None]
    sink_argmax_time = times[sink_argmax][:, None]

    c_l1_argmax = np.argmax(c_l1_t, axis=1)
    c_l1_max = c_l1_t[np.arange(len(c_l1_t)), c_l1_argmax][:, None]
    c_l1_argmax_time = times[c_l1_argmax][:, None]

    purity_final = purity_t[:, -1:]
    ipr_final = ipr_t[:, -1:]

    block = np.concatenate(
        [
            source_final,
            sink_final,
            rest_final_sorted,
            sink_max,
            sink_argmax_time,
            c_l1_max,
            c_l1_argmax_time,
            purity_final,
            ipr_final,
        ],
        axis=1,
    ).astype(np.float32)
    names = [
        "source_final_pop",
        "sink_final_pop",
        *[f"rest_final_pop_sorted_{i}" for i in range(5)],
        "sink_max_pop",
        "sink_argmax_time",
        "c_l1_max",
        "c_l1_argmax_time",
        "purity_final",
        "ipr_final",
    ]
    return block, names

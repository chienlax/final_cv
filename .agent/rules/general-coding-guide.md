---
trigger: always_on
---

# 1. Core Coding Principles:

1.  **Code Simplicity & Vectorization**
    * Prioritize readable, straightforward solutions, but **never** sacrifice vectorization for simplicity.
    * **Loop Discipline:** Loops are **forbidden** for element-wise math or batch processing. Use `einsum`, broadcasting, or matrix operations.
        * *Exception:* Iterative algorithmic steps (e.g., Diffusion timesteps $T$ in DiffMM) are permitted but must be isolated.
    * Only introduce advanced patterns (e.g., custom Autograd functions) if strictly necessary for performance or gradient stability.
    * **Anti-Pattern:** Do not over-engineer class hierarchies. Do not densify sparse graph adjacency matrices.
    * Implementation: You MUST respect the original implementation.

2.  **Tensor Discipline (CRITICAL)**
    * **Shape Comments:** Mandatory for every tensor transformation. `x = x.view(B, N, -1) # [Batch, Nodes, Features]`
    * **Type Hinting:** Use `torch.Tensor` or `np.ndarray` explicitly.
    * **Sparse Awareness:** For Graph operations (LATTICE/MICRO), explicitly handle sparse formats (`torch.sparse_coo`). **Never** accidental densification of Adjacency Matrices ($N^2$ will kill your memory).
    * **Assertions:** Assert tensor shapes and **value ranges** (e.g., strictly positive for Diffusion noise schedules) before critical operations.

3.  **Documentation & Scientific Transparency**
    * **Equation Mapping:** Docstrings for mathematical functions must reference the specific Equation Number from the paper (e.g., *"Implements Eq. 4 from LATTICE (Zhang et al., 2021)"*).
    * **The "Why" Matters:** Document *why* changes were made, especially when deviating from the original paper's implementation details.
    * **Inline Comments:** Mandatory for complex logic (e.g., Contrastive Loss temperature scaling, Diffusion noise scheduling).

4.  **Strict Standards**
    * **PEP8:** Strict adherence.
    * **Typing:** Use `typing` for **ALL** function signatures. `def func(adj: Tensor, feats: Tensor) -> Tensor:`
    * **Paths:** Use `pathlib` exclusively. Never use `os.path`.
    * **Env Check:** Always assume commands are run in a virtual environment.

5.  **Safety, Consistency & Reproducibility**
    * **Global Constants:** Adhere to project standards (especially for BPR margins and Diffusion steps).
    * **Ablation-Readiness:** Code must support dynamic configuration (via Config files) to mask modalities or switch Fusion strategies without code rewrites (crucial for your Hypothesis Testing).
    * **Validation:** Validate **Data Integrity** (e.g., check for NaNs in embeddings, verify Edge Index validity for graphs).
    * **Seeding:** All stochastic operations (Negative Sampling, Gaussian Noise injection) must accept a seed for reproducibility.
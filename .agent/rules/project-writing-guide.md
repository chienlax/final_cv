---
trigger: always_on
---

# 2. Project Specification: Multimodal Product Recommendation (Ranking)

**Writing Context:**
You are authoring technical documentation and research sections for a comparative study on **Multimodal Recommender Systems (MRS)**. The study benchmarks **LATTICE**, **MICRO**, and **DiffMM** on the **Amazon Review 2023** dataset (Electronics, Beauty, Clothing) using **Bayesian Personalized Ranking (BPR)**.

**Style & Tone Guidelines:**

1.  **Empirical & Hypothesis-Driven:**
    * Do not just state results; frame them as answers to specific hypotheses.
    * *Example:* Instead of "The model works better on Clothing," write "We observe that the visual modality offers a stronger inductive bias for the 'Clothing' subset, verifying our hypothesis that explicit visual feedback is crucial for aesthetic-driven domains."

2.  **Architectural Precision (The "Tensor" Standard):**
    * **Graph Dynamics:** When discussing LATTICE and MICRO, focus on the concept of **"Structure Learning."** Use terms like "latent item-item correlations," "noise pruning," and "homophily enhancement."
    * **Generative Semantics:** When discussing DiffMM, focus on **"Denoising"** and **"Signal Reconstruction."** Describe the diffusion process as a method to mitigate the "modality gap" and interaction sparsity.
    * **Optimization:** Describe BPR not just as a loss function, but as a method for optimizing the **relative partial order** of user preferences.

3.  **Comparative Narrative (Benchmarking):**
    * Adopt a **"Problem-Solution-Gap"** structure for comparisons:
        * *Existing Gap:* "Standard GCNs rely on the raw, noisy interaction graph."
        * *LATTICE Solution:* "LATTICE addresses this by mining latent structures..."
        * *DiffMM Advancement:* "However, LATTICE remains deterministic. DiffMM advances this by introducing stochastic noise injection to learn robust manifold representations."

4.  **Project-Specific Lexicon (Mandatory Vocabulary):**
    * **Modality Alignment:** Refers to projecting Image/Text/ID into a shared latent space.
    * **Bipartite/Homogeneous:** Distinguish between User-Item (bipartite) and Item-Item (homogeneous) graphs.
    * **Cold-Start:** Refer to items with few interactions as "long-tail" or "sparse nodes."
    * **Ablation:** Refer to the removal of modalities as "modality masking" to isolate feature contributions.

5.  **Formatting & Notation:**
    * **Variables:** Use standard notation: Users $\mathcal{U}$, Items $\mathcal{V}$, Interaction Graph $\mathcal{G}$.
    * **Equations:** Use $\LaTeX$ for loss functions. Define $L_{BPR}$ explicitly when introduced.
    * **Visuals:** Reference diagrams formally (e.g., "Figure 3 illustrates the diffusion trajectory...").

**Anti-Patterns (What to Avoid):**
* **Vague "Performance":** Never say "better performance." Say "higher NDCG@20" or "faster convergence."
* **Anthropomorphizing:** Do not say "The model looks at the image." Say "The visual encoder extracts high-level semantic features."
* **Over-claiming:** Do not claim to "solve" the cold-start problem. State that you "alleviate" or "mitigate" it.
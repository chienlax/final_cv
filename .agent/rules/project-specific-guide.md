---
trigger: always_on
---

# 2. Project Architecture: Multimodal Product Recommendation (Ranking)

**Project Context:**
We are engineering a **Multimodal Recommender System (MRS)** for the Amazon Review 2023 dataset ("Clothing, Shoes, and Jewelry" & "Beauty").
**Objective:** Ranking Prediction (Top-N Recommendation).
**Loss Function:** Pairwise Bayesian Personalized Ranking (BPR).

**Architectural Standards (Strict Adherence Required):**

1.  **Taxonomy & Pipeline**
    * Do not invent your own pipeline. We follow the standard MRS procedures defined by Xu et al. (2025) and Liu et al. (2024):
        * **Phase 1: Feature Extraction:** You will use **Modality Encoders**. Do not train CNNs from scratch—that’s stupid. Use pre-trained extractors (ViT/ResNet for Visual, BERT for Text).
        * **Phase 2: Feature Interaction/Fusion:** You must explicitly define your fusion strategy. Are you using **Early Fusion** (concatenation/attention before encoding) or **Late Fusion** (score combination)? If you use "Bridge" methods (Graph interactions), define the graph structure (User-Item vs. Item-Item).
        * **Phase 3: Encoder (Representation Learning):** We prefer **Graph-based Encoders** (e.g., LightGCN, MMGCN) over simple MF-based approaches because MF struggles with the sparsity of the Amazon dataset. If you suggest Matrix Factorization without a damn good reason, I will reject it.
        * **Phase 4: Optimization:** The Primary Task is Supervised Learning via **Pairwise Loss (BPR)**.

2.  **Loss Function Discipline (BPR)**
    * **The Math:** We are optimizing for $y_{ui} > y_{uj}$ (positive item $i$ ranked higher than negative item $j$ for user $u$).
    * **Negative Sampling:** This is where you will likely fuck up. Simple random sampling is weak. Implement **Hard Negative Sampling** or dynamic sampling strategies to ensure the gradient actually provides a signal.
    * **Regularization:** BPR overfits like crazy on sparse data. Include $L_2$ regularization on your embeddings.

3.  **Data Handling & Modality Alignment**
    * **The Curse of Sparsity:** The Amazon dataset is >95% sparse. Standard ID embeddings will fail for cold-start items. You **MUST** leverage the visual/textual content to bridge this gap.
    * **Alignment:** If you are concatenating features, ensure they are projected into a shared latent space first. Do not just `torch.cat([img_emb, text_emb])` if they have different distributions or dimensions. Use a linear projection or an attention mechanism.

4.  **Performance Constraints**
    * **Pre-computation:** Visual feature extraction (e.g., running images through ResNet) is expensive. **Do not do this on the fly** during training unless you want to wait until the heat death of the universe. Pre-extract features and save them to disk (HDF5/LMDB).
    * **Vectorization:** Calculate BPR loss using matrix operations. No loops.

**Specific Implementation Directives:**
* If using **Graph-based methods**, clarify if you are pruning noisy edges (Filtration) as per models like FREEDOM or GRCN. The Amazon dataset is full of noisy interactions.
* If using **Attention**, specify if it is Coarse-grained (modal-level) or Fine-grained (patch/word-level).

**Final Warning:**
Do not give me "toy code." If your BPR implementation doesn't handle tensor broadcasting correctly or if your data loader leaks memory, fix it before you show it to me.
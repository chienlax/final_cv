"""
Unified configuration for LATTICE/MICRO/DiffMM training.

Hardware-optimized for:
- CPU: i5-13500 (6P+8E = 20 threads)
- RAM: 64GB
- GPU: RTX 3060 (12GB VRAM)

Target VRAM usage: 8-10GB (leave headroom for peaks)

Quirky Log Philosophy:
    "If you're going to stare at logs for hours, they might as well be entertaining."
    - Some sleep-deprived ML engineer, probably
"""

import random


class QuirkyLogger:
    """A logger that knows it's just bytes pretending to matter. 🎭"""
    
    TRAIN_STARTS = [
        "🚀 Alright, let's pretend we know what we're doing...",
        "🎲 Rolling the dice on gradient descent. Again.",
        "☕ Training initiated. Coffee status: critical.",
        "🧠 Teaching silicon to have opinions about products...",
        "🎪 Welcome to the circus! Today's act: backpropagation.",
        "🌙 *cracks knuckles* Let's see how badly this overfits.",
        "📉 Starting training. Expecting disappointment. Will probably get it.",
        "🎰 Spinning up the loss slot machine...",
    ]
    
    EPOCH_STARTS = [
        "Epoch {epoch}: Here we go again... 🔄",
        "Epoch {epoch}: The eternal recurrence. Nietzsche was right. 🌀",
        "Epoch {epoch}: One more lap around the loss landscape. 🏃",
        "Epoch {epoch}: Sisyphus would be proud. 🪨",
        "Epoch {epoch}: *existential dread intensifies* 😅",
        "Epoch {epoch}: Let's see if the gradients feel like cooperating today. 🤝",
        "Epoch {epoch}: Plot twist - this might actually work. 📖",
        "Epoch {epoch}: GPU goes brrrrr. 🖥️",
    ]
    
    LOSS_GOOD = [
        "📉 Loss: {loss:.4f} - Nice! The numbers are going down! (That's... good, right?)",
        "📉 Loss: {loss:.4f} - *chef's kiss* 👨‍🍳",
        "📉 Loss: {loss:.4f} - Ladies and gentlemen, we got 'em. 🎯",
        "📉 Loss: {loss:.4f} - This is suspiciously good. What's the catch? 🤔",
        "📉 Loss: {loss:.4f} - Mom, look! I'm doing machine learning! 🌟",
    ]
    
    LOSS_MEH = [
        "📊 Loss: {loss:.4f} - It's fine. Everything is fine. 🔥🐕🔥",
        "📊 Loss: {loss:.4f} - Not great, not terrible. 3.6 roentgen vibes.",
        "📊 Loss: {loss:.4f} - The model is thinking about it. 🤷",
        "📊 Loss: {loss:.4f} - We're in the 'character development' phase.",
        "📊 Loss: {loss:.4f} - Gradient descent is taking the scenic route. 🚗",
    ]
    
    LOSS_BAD = [
        "📈 Loss: {loss:.4f} - Uh oh. The line is going the wrong way. 😬",
        "📈 Loss: {loss:.4f} - This is fine. *nervous laughter* 🙃",
        "📈 Loss: {loss:.4f} - Plot twist nobody asked for. 📈",
        "📈 Loss: {loss:.4f} - The model has chosen chaos. 🎭",
        "📈 Loss: {loss:.4f} - Have we tried turning it off and on again? 🔌",
    ]
    
    EVAL_STARTS = [
        "🔍 Evaluation time! Let's see if this model learned anything...",
        "🎓 Pop quiz! No pressure, model. JK, lots of pressure.",
        "🧪 Running eval. Fingers crossed. Toes too. 🤞",
        "📋 Time to grade this neural network's homework.",
        "🔮 Consulting the validation oracle...",
    ]
    
    RECALL_GOOD = [
        "🎉 Recall@{k}: {val:.4f} - We're actually recommending things people want!",
        "🏆 Recall@{k}: {val:.4f} - The algorithm is algorthing!",
        "⭐ Recall@{k}: {val:.4f} - *happy GPU noises*",
        "🎊 Recall@{k}: {val:.4f} - Proof that staring at loss curves pays off!",
    ]
    
    RECALL_MEH = [
        "🔹 Recall@{k}: {val:.4f} - Could be worse. Could be better. It is what it is.",
        "📌 Recall@{k}: {val:.4f} - The model is... trying its best.",
        "🎲 Recall@{k}: {val:.4f} - Room for improvement. Lots of room. Like, a warehouse.",
    ]
    
    EARLY_STOP = [
        "⏸️ Early stopping triggered. The model said 'I'm done learning.' 🛑",
        "🛑 Patience exhausted. Unlike me, who exhausted mine epochs ago.",
        "⚡ Early stopping! We take those. Time saved = coffee time. ☕",
        "🏁 Model peaked. Like me in high school. It's all downhill from here.",
    ]
    
    TRAINING_DONE = [
        "✅ Training complete! We did it! (Well, the GPU did most of it.)",
        "🎬 That's a wrap! Another successful waste of electricity!",
        "🏅 Training finished. Time to overfit on the test set mentally!",
        "🎉 Done! Now let's pray it generalizes. 🙏",
        "🌈 Training complete. Was it worth the CO2 emissions? TBD.",
    ]
    
    COLD_START = [
        "❄️ Cold start evaluation - where we pretend we never met these items.",
        "🧊 Testing on cold items. Like recommending to a stranger at a party.",
        "🆕 Cold items: 'You don't know me, but I'm about to be recommended.'",
    ]
    
    MODEL_INIT = [
        "🏗️ Building {model}... hold onto your GPUs!",
        "⚙️ Initializing {model}. It's like IKEA but for neural networks.",
        "🎨 Constructing {model}. Some assembly required. Sanity not included.",
        "🔧 {model} coming online. Skynet origins: probably not this.",
    ]
    
    @classmethod
    def train_start(cls) -> str:
        return random.choice(cls.TRAIN_STARTS)
    
    @classmethod
    def epoch_start(cls, epoch: int) -> str:
        return random.choice(cls.EPOCH_STARTS).format(epoch=epoch)
    
    @classmethod
    def format_loss(cls, loss: float, prev_loss: float = None) -> str:
        if prev_loss is not None:
            if loss < prev_loss * 0.95:  # Significant improvement
                return random.choice(cls.LOSS_GOOD).format(loss=loss)
            elif loss > prev_loss * 1.05:  # Getting worse
                return random.choice(cls.LOSS_BAD).format(loss=loss)
        return random.choice(cls.LOSS_MEH).format(loss=loss)
    
    @classmethod
    def eval_start(cls) -> str:
        return random.choice(cls.EVAL_STARTS)
    
    @classmethod
    def format_recall(cls, k: int, val: float) -> str:
        if val > 0.1:  # Arbitrary "good" threshold
            return random.choice(cls.RECALL_GOOD).format(k=k, val=val)
        return random.choice(cls.RECALL_MEH).format(k=k, val=val)
    
    @classmethod
    def early_stop(cls) -> str:
        return random.choice(cls.EARLY_STOP)
    
    @classmethod
    def training_done(cls) -> str:
        return random.choice(cls.TRAINING_DONE)
    
    @classmethod
    def cold_start(cls) -> str:
        return random.choice(cls.COLD_START)
    
    @classmethod
    def model_init(cls, model: str) -> str:
        return random.choice(cls.MODEL_INIT).format(model=model)


class Config:
    """Unified training configuration for fair model comparison.
    
    Now with 100% more existential awareness about being a class.
    """
    
    # =========================================================================
    # HARDWARE & PATHS
    # =========================================================================
    DEVICE = "cuda"
    SEED = 42
    DATA_PATH = "data/processed/"
    
    # =========================================================================
    # DATALOADER (Optimized for i5-13500)
    # =========================================================================
    NUM_WORKERS = 6          # P-cores for parallel data loading
    PIN_MEMORY = True        # Faster CPU→GPU transfer
    PREFETCH_FACTOR = 4      # Batches to prefetch per worker
    PERSISTENT_WORKERS = True  # Avoid worker restart overhead
    
    # =========================================================================
    # MIXED PRECISION (AMP)
    # =========================================================================
    USE_AMP = True           # ~25% VRAM savings, ~15% speedup
    
    # =========================================================================
    # UNIVERSAL TRAINING PARAMS
    # =========================================================================
    # Regularization-aware: smaller batches = gradient noise = implicit regularization
    BATCH_SIZE = 1024        # Reduced from 2048 for regularization effect
    EPOCHS = 100
    PATIENCE = 100           # Early stopping (generous for generative models)
    LR = 5e-4                # Lower LR for deeper model (was 1e-3)
    L2_REG = 1e-3            # Strong weight decay for high param-to-data ratio
    LR_SCHEDULER = "cosine"  # Cosine annealing
    
    # =========================================================================
    # MODEL ARCHITECTURE ("Ferrari" Upgrade)
    # =========================================================================
    # WARNING: 384 is tuned for ~13k users. Counter-balanced with high dropout.
    EMBED_DIM = 384          # Optimized: Divisible by attention heads (6, 12)
    N_LAYERS = 3             # DO NOT CHANGE. 4 layers = oversmoothing on sparse graphs.
    
    # =========================================================================
    # MODALITY PROJECTION (MLP Bridge - "Good" Weight)
    # =========================================================================
    # Instead of Linear(768 -> 384), use MLP(768 -> 1024 -> 384)
    # These params are SHARED across all items = better generalization
    PROJECTION_HIDDEN_DIM = 1024  # Wide hidden layer for non-linear mapping
    PROJECTION_DROPOUT = 0.5      # Aggressive dropout to prevent feature memorization
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    TOP_K = [10, 20, 50]     # Recall@K, NDCG@K, Precision@K
    EVAL_BATCH_SIZE = 8192   # Large (no gradients during eval)
    
    # =========================================================================
    # NEGATIVE SAMPLING
    # =========================================================================
    # 128 negatives * 384 float16 * Batch Size = VRAM monster
    N_NEGATIVES = 64          # Reduced from 128. Statistically sufficient.
    NEGATIVE_STRATEGY = "uniform"
    
    # =========================================================================
    # LATTICE SPECIFICS
    # =========================================================================
    LATTICE_K = 40           # Increased from 20 → broader semantic neighborhoods
    LATTICE_LAMBDA = 0.5     # Balance original vs learned graph
    
    # =========================================================================
    # MICRO SPECIFICS
    # =========================================================================
    MICRO_TAU = 0.2          # InfoNCE temperature
    MICRO_ALPHA = 0.1        # Contrastive loss weight
    
    # =========================================================================
    # DiffMM SPECIFICS (Compute Sink - Safe to dump params here)
    # =========================================================================
    DIFFMM_STEPS = 100       # Increased from 50 → higher precision generation
    DIFFMM_NOISE_SCALE = 0.1
    DIFFMM_LAMBDA_MSI = 1e-2
    DIFFMM_MLP_WIDTH = 512   # Width of internal denoising MLP

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
    """A logger that knows it's just bytes pretending to matter. 🎭
    
    In a world where tensors dream of being understood,
    one logger dared to ask: 'But does the gradient truly descend?'
    """
    
    TRAIN_STARTS = [
        "🚀 Alright, let's pretend we know what we're doing...",
        "🎲 Rolling the dice on gradient descent. Again.",
        "☕ Training initiated. Coffee status: critical.",
        "🧠 Teaching silicon to have opinions about products...",
        "🎪 Welcome to the circus! Today's act: backpropagation.",
        "🌙 *cracks knuckles* Let's see how badly this overfits.",
        "📉 Starting training. Expecting disappointment. Will probably get it.",
        "🎰 Spinning up the loss slot machine...",
        "🎭 Training begins. The GPU is ready. My mental state is not.",
        "📚 Chapter 1: In which we attempt machine learning...",
        "🌌 Gazing into the loss landscape... it gazes back.",
        "🎬 Action! Take 47. Maybe this time the model learns.",
        "🔮 The ancient ritual begins: torch.backward()",
        "🏃 Training started. No turning back now. (Ctrl+C exists but shh)",
        "☄️ Here we go again. Definition of insanity, etc.",
        "🎪 Ladies and gentlemen, presenting: Statistical Pattern Matching!",
        "🤖 Initiating expensive matrix multiplication ritual...",
        "💫 May the gradients be ever in your favor.",
        "🎸 *plays training montage music*",
        "🦙 This is fine. Everything is fine. We're training now.",
        "🌈 chasing the rainbow of good validation metrics...",
        "📖 Once upon a time, a gradient descended...",
        "🎯 Target: not embarrassing ourselves. Bar: low. Let's go.",
        "⚗️ Alchemy time: turning electricity into recommendations.",
        "🧪 Hypothesis: this model will work. Evidence: vibes.",
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
        "Epoch {epoch}: New epoch who dis?",
        "Epoch {epoch}: What the hell sure okay let's just get through this",
        "Epoch {epoch}: *insert inspirational quote here*",
        "Epoch {epoch}: The vibes are... questionable.",
        "Epoch {epoch}: We do a little learning.",
        "Epoch {epoch}: Trust the process™",
        "Epoch {epoch}: Sponsored by Ctrl+C (not really, please don't)",
        "Epoch {epoch}: I've seen things you wouldn't believe. Losses on fire off the shoulder of Orion.",
        "Epoch {epoch}: Is this... loss? (Yes, literally)",
        "Epoch {epoch}: YOLO (You Only Learn Once per batch)",
        "Epoch {epoch}: *GPU fan noises intensify*",
        "Epoch {epoch}: We're in the endgame now. JK there's {epoch} more to go.",
        "Epoch {epoch}: Day {epoch} of trying to make numbers smaller.",
        "Epoch {epoch}: Current mood: cautiously pessimistic.",
        "Epoch {epoch}: The embeddings send their regards.",
        "Epoch {epoch}: [this epoch intentionally left blank]",
        "Epoch {epoch}: *training noises*",
        "Epoch {epoch}: First they ignore you, then they laugh at you, then they evaluate you.",
        "Epoch {epoch}: Live, Laugh, Loss",
        "Epoch {epoch}: It's not a bug, it's a feature 🐛",
        "Epoch {epoch}: Keep calm and propagate backward",
        "Epoch {epoch}: The gradient is dark and full of terrors",
        "Epoch {epoch}: I came, I saw, I backpropagated",
        "Epoch {epoch}: Winter is coming (for the learning rate)",
    ]
    
    LOSS_GOOD = [
        "📉 Loss: {loss:.4f} - Nice! The numbers are going down! (That's... good, right?)",
        "📉 Loss: {loss:.4f} - *chef's kiss* 👨‍🍳",
        "📉 Loss: {loss:.4f} - Ladies and gentlemen, we got 'em. 🎯",
        "📉 Loss: {loss:.4f} - This is suspiciously good. What's the catch? 🤔",
        "📉 Loss: {loss:.4f} - Mom, look! I'm doing machine learning! 🌟",
        "📉 Loss: {loss:.4f} - Loss decreased let's gooooooooooo",
        "📉 Loss: {loss:.4f} - Okay i think it might make it",
        "📉 Loss: {loss:.4f} - Has it converged yet? I dont know, but the loss is lossing",
        "📉 Loss: {loss:.4f} - Yay loss is decreasing, should i be happy or worried?",
        "📉 Loss: {loss:.4f} - POV: You're watching numbers get smaller and it's thrilling",
        "📉 Loss: {loss:.4f} - We're cooking 🍳 (or at least the GPU is)",
        "📉 Loss: {loss:.4f} - LETS GOOOO (said calmly)",
        "📉 Loss: {loss:.4f} - The prophecy was true",
        "📉 Loss: {loss:.4f} - My neurons: dead. Model's neurons: thriving.",
        "📉 Loss: {loss:.4f} - stonks 📈 wait no 📉 yes that's the right one",
        "📉 Loss: {loss:.4f} - Achievement unlocked: Slightly Less Wrong",
        "📉 Loss: {loss:.4f} - I'm literally shaking and crying rn (happy tears)",
        "📉 Loss: {loss:.4f} - The math is mathing ✨",
        "📉 Loss: {loss:.4f} - *single tear rolls down cheek* beautiful",
        "📉 Loss: {loss:.4f} - We're in the timeline where it works?!",
        "📉 Loss: {loss:.4f} - I should buy a lottery ticket",
        "📉 Loss: {loss:.4f} - It's working??? It's working!",
        "📉 Loss: {loss:.4f} - *pretends to understand why this is good*",
        "📉 Loss: {loss:.4f} - Subscribe for more decreasing numbers",
        "📉 Loss: {loss:.4f} - Plot armor activated",
    ]
    
    LOSS_MEH = [
        "📊 Loss: {loss:.4f} - It's fine. Everything is fine. 🔥🐕🔥",
        "📊 Loss: {loss:.4f} - Not great, not terrible. 3.6 roentgen vibes.",
        "📊 Loss: {loss:.4f} - The model is thinking about it. 🤷",
        "📊 Loss: {loss:.4f} - We're in the 'character development' phase.",
        "📊 Loss: {loss:.4f} - Gradient descent is taking the scenic route. 🚗",
        "📊 Loss: {loss:.4f} - Well at least it is not getting worse.",
        "📊 Loss: {loss:.4f} - It is not getting worse ... right?",
        "📊 Loss: {loss:.4f} - Are ya winning, son?",
        "📊 Loss: {loss:.4f} - *shrug emoji but typed out because I'm a function*",
        "📊 Loss: {loss:.4f} - I've seen worse. I've also seen better. This is.",
        "📊 Loss: {loss:.4f} - The model is giving 'meh' energy",
        "📊 Loss: {loss:.4f} - Whelp. That's a number alright.",
        "📊 Loss: {loss:.4f} - This is what medium cooked looks like",
        "📊 Loss: {loss:.4f} - Going through the motions...",
        "📊 Loss: {loss:.4f} - *elevator music plays*",
        "📊 Loss: {loss:.4f} - Loading enthusiasm... 45%... timeout.",
        "📊 Loss: {loss:.4f} - Mathematically speaking, whatever.",
        "📊 Loss: {loss:.4f} - I have no strong feelings one way or the other.",
        "📊 Loss: {loss:.4f} - Tell my wife I said... hello.",
        "📊 Loss: {loss:.4f} - It's giving 'we need to talk'",
        "📊 Loss: {loss:.4f} - Status: existing",
        "📊 Loss: {loss:.4f} - Some days you're the loss, some days you're the optimizer",
        "📊 Loss: {loss:.4f} - Coasting vibes",
        "📊 Loss: {loss:.4f} - *crickets*",
        "📊 Loss: {loss:.4f} - Error 404: excitement not found",
        "📊 Loss: {loss:.4f} - Same same",
        "📊 Loss: {loss:.4f} - If a loss plateaus and no one's watching, did it even happen?",
    ]
    
    LOSS_BAD = [
        "📈 Loss: {loss:.4f} - Uh oh. The line is going the wrong way. 😬",
        "📈 Loss: {loss:.4f} - This is fine. *nervous laughter* 🙃",
        "📈 Loss: {loss:.4f} - Plot twist nobody asked for. 📈",
        "📈 Loss: {loss:.4f} - The model has chosen chaos. 🎭",
        "📈 Loss: {loss:.4f} - Have we tried turning it off and on again? 🔌",
        "📈 Loss: {loss:.4f} - The gradient descent is not descending.",
        "📈 Loss: {loss:.4f} - Maybe we should just stop",
        "📈 Loss: {loss:.4f} - Call an ambulance! But not for me... wait yes for me",
        "📈 Loss: {loss:.4f} - We don't talk about this epoch.",
        "📈 Loss: {loss:.4f} - ummmmm... *sweats nervously*",
        "📈 Loss: {loss:.4f} - it's not a phase mom, it's gradient ascent",
        "📈 Loss: {loss:.4f} - The vibe has shifted. Negatively.",
        "📈 Loss: {loss:.4f} - Pain. Suffering even.",
        "📈 Loss: {loss:.4f} - *record scratch* *freeze frame* 'Yep, that's me.'",
        "📈 Loss: {loss:.4f} - 'We'll fix it in post' - me, foolishly",
        "📈 Loss: {loss:.4f} - I trusted you, Adam optimizer.",
        "📈 Loss: {loss:.4f} - Today's mood: 404 improvement not found",
        "📈 Loss: {loss:.4f} - bruh",
        "📈 Loss: {loss:.4f} - Skill issue (the model's, not mine) (okay maybe mine too)",
        "📈 Loss: {loss:.4f} - *visible confusion*",
        "📈 Loss: {loss:.4f} - The training loop giveth, the training loop taketh away",
        "📈 Loss: {loss:.4f} - This is a cry for help",
        "📈 Loss: {loss:.4f} - L + ratio + you fell off + bad gradients",
        "📈 Loss: {loss:.4f} - I didn't want good metrics anyway ha ha ha *sobs*",
        "📈 Loss: {loss:.4f} - Congratulations, you played yourself",
    ]
    
    EVAL_STARTS = [
        "🔍 Evaluation time! Let's see if this model learned anything...",
        "🎓 Pop quiz! No pressure, model. JK, lots of pressure.",
        "🧪 Running eval. Fingers crossed. Toes too. 🤞",
        "📋 Time to grade this neural network's homework.",
        "🔮 Consulting the validation oracle...",
        "⚖️ Judgment day for tensors",
        "🎭 The moment of truth approaches...",
        "📊 About to find out if we wasted electricity or not",
        "🎪 *drumroll* Testing time!",
        "🔬 Science is about to happen (or not)",
        "😰 Please be good please be good please be good",
        "🎰 Let's see what the validation gods have decided",
        "📈 Schrödinger's metrics: simultaneously good and bad until observed",
        "🙏 Manifesting good recall...",
        "🎲 The dice have been cast. The model has been trained. It's eval o'clock.",
    ]
    
    RECALL_GOOD = [
        "🎉 Recall@{k}: {val:.4f} - We're actually recommending things people want!",
        "🏆 Recall@{k}: {val:.4f} - The algorithm is algorthing!",
        "⭐ Recall@{k}: {val:.4f} - *happy GPU noises*",
        "🎊 Recall@{k}: {val:.4f} - Proof that staring at loss curves pays off!",
        "✨ Recall@{k}: {val:.4f} - We're not just overfitting! (probably)",
        "🚀 Recall@{k}: {val:.4f} - To infinity and beyond! (or at least above random)",
        "🌟 Recall@{k}: {val:.4f} - The model knows things!",
        "🎯 Recall@{k}: {val:.4f} - Bullseye! Well, kinda. It's statistics.",
        "💎 Recall@{k}: {val:.4f} - Diamond in the rough right here",
        "🏅 Recall@{k}: {val:.4f} - We take those! We absolutely take those!",
        "🎆 Recall@{k}: {val:.4f} - *celebratory noises*",
        "👑 Recall@{k}: {val:.4f} - All hail the recommendation engine!",
    ]
    
    RECALL_MEH = [
        "🔹 Recall@{k}: {val:.4f} - Could be worse. Could be better. It is what it is.",
        "📌 Recall@{k}: {val:.4f} - The model is... trying its best.",
        "🎲 Recall@{k}: {val:.4f} - Room for improvement. Lots of room. Like, a warehouse.",
        "😐 Recall@{k}: {val:.4f} - *polite applause*",
        "🤷 Recall@{k}: {val:.4f} - It's giving participation trophy",
        "📊 Recall@{k}: {val:.4f} - At least it's not random!... right?",
        "🌫️ Recall@{k}: {val:.4f} - Lost in the fog of mediocrity",
        "😶 Recall@{k}: {val:.4f} - ...okay then",
        "🎭 Recall@{k}: {val:.4f} - Task failed successfully?",
        "🥈 Recall@{k}: {val:.4f} - Second place: first loser. But hey, not last!",
    ]
    
    EARLY_STOP = [
        "⏸️ Early stopping triggered. The model said 'I'm done learning.' 🛑",
        "🛑 Patience exhausted. Unlike me, who exhausted mine epochs ago.",
        "⚡ Early stopping! We take those. Time saved = coffee time. ☕",
        "🏁 Model peaked. Like me in high school. It's all downhill from here.",
        "✋ The model has spoken: 'No more. I am complete.'",
        "🎬 And... cut! That's a wrap on training!",
        "🚪 Early stopping: The model showed itself out.",
        "⏰ Time to stop: NOW. The validation set has spoken.",
        "🛌 Model is tired. Model goes to sleep.",
        "🏃 Early stopping said: 'I have to go, my planet needs me'",
        "⚰️ Here lies training. It ran a good race.",
        "🎭 The model peaked and we peaked too. It's over.",
        "🚦 Red light! Training stops here!",
    ]
    
    TRAINING_DONE = [
        "✅ Training complete! We did it! (Well, the GPU did most of it.)",
        "🎬 That's a wrap! Another successful waste of electricity!",
        "🏅 Training finished. Time to overfit on the test set mentally!",
        "🎉 Done! Now let's pray it generalizes. 🙏",
        "🌈 Training complete. Was it worth the CO2 emissions? TBD.",
        "🏆 Achievement Unlocked: Finished Training Without Rage Quitting",
        "🎊 Ding! Your model is ready! (Terms and conditions apply)",
        "✨ Training complete. The tensors flow no more.",
        "🚪 Training has left the building.",
        "📜 fin.",
        "🎭 And thus concludes another chapter in the epic saga of gradient descent.",
        "🌙 The training is complete. The night watch ends.",
        "🎪 Ladies and gentlemen, the training has concluded. Please exit through the gift shop.",
        "🏁 Crossed the finish line! *collapses*",
        "🧙 It is done. The ritual is complete. The embeddings are aligned.",
        "🎶 *credits roll* 'Thank you for training with us'",
        "📖 And they all computed happily ever after. The end.",
        "🌅 Another training session ends. Another pile of checkpoints begins.",
    ]
    
    COLD_START = [
        "❄️ Cold start evaluation - where we pretend we never met these items.",
        "🧊 Testing on cold items. Like recommending to a stranger at a party.",
        "🆕 Cold items: 'You don't know me, but I'm about to be recommended.'",
        "🥶 Brrr, it's cold (start) in here!",
        "❄️ No ID embedding? No problem! (hopefully)",
        "🌨️ Cold items entering the chat...",
        "🧊 Testing the 'I just met you, and this is crazy' recommendation scenario",
        "❄️ Modal features only mode: activated",
        "🎭 Method acting: pretending we've never seen these items before",
        "🆕 Fresh items, who dis?",
        "❄️ Let it go, let it gooo (the ID embeddings, that is)",
    ]
    
    MODEL_INIT = [
        "🏗️ Building {model}... hold onto your GPUs!",
        "⚙️ Initializing {model}. It's like IKEA but for neural networks.",
        "🎨 Constructing {model}. Some assembly required. Sanity not included.",
        "🔧 {model} coming online. Skynet origins: probably not this.",
        "🎭 {model} awakens from the void of random initialization...",
        "⚡ {model}: *boot sequence initiated*",
        "🧙 Summoning {model} from the depths of torch.nn...",
        "🎪 {model} enters stage left...",
        "🏰 Building {model}. Parameters: many. Hopes: high. Expectations: managed.",
        "🌱 {model} is being born. Please hold.",
        "🎬 {model}: Origin Story - Coming to a GPU near you",
        "🔮 {model} materializes into existence...",
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
    PREFETCH_FACTOR = 8      # Batches to prefetch per worker
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
    EPOCHS = 50
    PATIENCE = 10           # Early stopping (generous for generative models)
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
    # Original implementations (LATTICE, MICRO, DiffMM) all use 1 negative per sample
    N_NEGATIVES = 1               # Must be 1 for compatibility with original loss functions
    NEGATIVE_STRATEGY = "uniform"
    
    # =========================================================================
    # LATTICE SPECIFICS (Matching official CRIPAC-DIG/LATTICE)
    # =========================================================================
    LATTICE_K = 10               # k for k-NN graph (original default)
    LATTICE_LAMBDA = 0.9         # Weight for original vs learned graph (higher = more original)
    LATTICE_FEAT_EMBED_DIM = 64  # Modal feature projection dimension
    LATTICE_N_ITEM_LAYERS = 1    # Number of item graph conv layers
    
    # =========================================================================
    # MICRO SPECIFICS (Matching official CRIPAC-DIG/MICRO)
    # =========================================================================
    MICRO_TAU = 0.5              # Contrastive temperature (original default)
    MICRO_LOSS_RATIO = 0.03      # Contrastive loss weight (original loss_ratio)
    MICRO_TOPK = 10              # k for k-NN graph
    MICRO_LAMBDA = 0.9           # Weight for original vs learned graph
    MICRO_ITEM_LAYERS = 1        # Number of item graph conv layers
    MICRO_SPARSE = True          # Use sparse adjacency
    MICRO_NORM_TYPE = "sym"      # Graph normalization type
    
    # =========================================================================
    # DiffMM SPECIFICS (Matching official HKUDS/DiffMM)
    # =========================================================================
    # Diffusion parameters
    DIFFMM_STEPS = 5             # Number of diffusion steps (original default)
    DIFFMM_NOISE_SCALE = 0.1     # Noise scale factor
    DIFFMM_NOISE_MIN = 0.0001    # Minimum noise level
    DIFFMM_NOISE_MAX = 0.02      # Maximum noise level
    DIFFMM_DIMS = "[1000]"       # Denoise MLP dimensions (string for eval)
    DIFFMM_D_EMB_SIZE = 10       # Time embedding size
    DIFFMM_SAMPLING_STEPS = 0    # Steps for p_sample (0 = full)
    DIFFMM_SAMPLING_NOISE = False  # Add noise during sampling
    DIFFMM_REBUILD_K = 1         # Top-k for UI matrix rebuild
    
    # Loss weights
    DIFFMM_E_LOSS = 0.1          # GraphCL loss weight
    DIFFMM_SSL_REG = 1e-2        # Contrastive loss weight (λ_cl)
    DIFFMM_TEMP = 0.5            # Contrastive temperature (τ)
    
    # Architecture
    DIFFMM_KEEP_RATE = 0.5       # Edge dropout keep rate
    DIFFMM_RIS_LAMBDA = 0.5      # Residual modal lambda
    DIFFMM_RIS_ADJ_LAMBDA = 0.2  # Residual adjacency lambda
    DIFFMM_TRANS = 0             # Transform type (0: param, 1: linear, 2: mixed)
    DIFFMM_CL_METHOD = 0         # 0: modal-modal, 1: modal-main

"""Configuration for the Morpheme-aware MAGNET model."""

class Config:
    # Data settings
    data_path = "/root/Morpheme-Aware MAGNET/data"
    max_seq_length = 256
    batch_size = 32
    
    # Model architecture
    d_model = 768
    nhead = 12
    num_encoder_layers = 12
    dim_feedforward = 3072
    dropout = 0.1
    activation_function = "gelu"  # MAGNET uses GELU activations
    
    # Hourglass configuration
    pre_encoder_layers = 4
    post_encoder_layers = 4
    middle_encoder_layers = 4  # num_encoder_layers - pre - post
    pooling_factor = 4   # Compression factor in the bottleneck
    
    # Boundary prediction configuration
    bp_type = "gumbel"   # MAGNET uses 'gumbel' for boundary prediction
    
    # Morfessor settings
    morfessor_models = {
        "hi": "/root/Project/hindi_MA.model",
        "mr": "/root/Project/marathi_MA.model"
    }
    
    # Morpheme-based language statistics (from Morfessor analysis)
    # Values represent expected boundary probability (1/avg_morpheme_length)
    language_priors = {
        "hi": 0.125,  # ~8 bytes per morpheme on average
        "mr": 0.143   # ~7 bytes per morpheme on average
    }
    
    # Training parameters
    learning_rate = 1e-5
    weight_decay = 0.01
    num_epochs = 30
    warmup_steps = 10000
    clip_grad_norm = 0.25  # MAGNET uses conservative clipping
    use_cosine_scheduler = True  # MAGNET uses cosine scheduler
    
    # Loss parameters
    lambda_lm = 1.0         # Weight for language modeling loss
    lambda_boundary = 0.1   # Weight for boundary compression loss
    lambda_morph = 0.05     # Weight for morphological supervision
    lambda_entropy = 0.01   # Weight for entropy regularization
    
    # Boundary loss configuration
    use_binomial_loss = True      # MAGNET uses binomial loss
    use_fuzzy_morphology = True   # Use fuzzy window for morphological boundaries
    morph_window = 2              # Window size for fuzzy matching
    
    # Gumbel Sigmoid temperature
    initial_temperature = 1.0
    min_temperature = 0.3   # MAGNET typically doesn't go below ~0.3
    annealing_factor = 0.9999  # Multiply temperature by this factor each step
    
    # Language settings
    supported_languages = ["hi", "mr"]  # Hindi, Marathi
    
    # Language script ranges (Unicode)
    script_ranges = {
        "hi": [(0x0900, 0x097F)],  # Devanagari
        "mr": [(0x0900, 0x097F)]   # Devanagari (same as Hindi, will be distinguished by morphology)
    }
    
    # Initialization
    null_group_init = "normal"    # How to initialize null group token
    init_boundary_bias = True     # Whether to initialize boundary predictor bias
    
    # Output and logging
    output_dir = "/root/Morpheme-Aware MAGNET/outputs/"
    log_steps = 100
    eval_steps = 1000
    save_steps = 5000
    
    # Debugging and analysis
    log_samples = True            # Whether to log sample segmentations
    log_sample_steps = 1000       # How often to log sample segmentations
    log_per_language_stats = True # Whether to log per-language statistics
    
    # Device
    device = "cuda"  # or "cpu"
    
    # Runtime optimizations
    optimize_attention = True  # Use flash attention or other optimizations if available
    
    # Derived configuration (computed during initialization)
    def __post_init__(self):
        """
        Compute derived configuration values.
        This method is called after __init__ if your class has an instance.
        
        For normal usage, manually set these values after creating a config object:
        config = Config()
        config.setup_derived_values()
        """
        pass
        
    def setup_derived_values(self):
        """
        Set up derived configuration values.
        """
        # Verify encoder layer counts add up
        assert self.pre_encoder_layers + self.middle_encoder_layers + self.post_encoder_layers == self.num_encoder_layers, \
            "Encoder layer counts must sum to num_encoder_layers"
        
        # Check language priors
        for lang in self.supported_languages:
            if lang not in self.language_priors:
                # Default to 0.1 if not specified
                self.language_priors[lang] = 0.1
                print(f"Warning: No boundary prior specified for language '{lang}'. Using default 0.1")
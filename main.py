"""Main script to run training or evaluation."""
import argparse
import os
import torch
from config import Config
from data import create_dataloaders
from model.magnet import EnhancedMAGNET 
from training.trainer import Trainer
from utils import set_seed, count_parameters
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Enhanced MAGNET model")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True, 
                      help="Run mode: 'train' or 'evaluate'")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (for evaluation)")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--lambda_boundary", type=float, default=None, help="Weight for boundary loss")
    parser.add_argument("--lambda_morph", type=float, default=None, help="Weight for morphological loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    return parser.parse_args()

def train(config):
    """Train the model using the provided configuration."""
    # Create data loaders
    train_loader, val_loader, _, morfessor_wrapper = create_dataloaders(config)
    
    print(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Initialize model
    model = EnhancedMAGNET(config)
    print(f"Model initialized with {count_parameters(model):,} trainable parameters")
    
    # Initialize trainer
    trainer = Trainer(model, config, train_loader, val_loader)
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(config.output_dir, "final_model.pt"))
    
    print("Training completed!")

def evaluate_model(config, model_path):
    """Evaluate the model using the provided configuration and model path."""
    # Create data loaders
    _, val_loader, test_loader, morfessor_wrapper = create_dataloaders(config)
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=config.device)
    model = EnhancedMAGNET(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    
    print(f"Loaded model from {model_path}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    val_metrics = evaluate.evaluate_model(model, val_loader, config)
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics = evaluate.evaluate_model(model, test_loader, config)
    
    # Print and save metrics
    print("\nValidation results:")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nTest results:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics to file
    results_file = os.path.join(config.output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("Validation results:\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\nTest results:\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Results saved to {results_file}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = Config()
    
    # Override config with command-line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.lambda_boundary:
        config.lambda_boundary = args.lambda_boundary
    if args.lambda_morph:
        config.lambda_morph = args.lambda_morph
    if args.device:
        config.device = args.device
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run in selected mode
    if args.mode == "train":
        train(config)
    elif args.mode == "evaluate":
        if not args.model_path:
            raise ValueError("Model path must be provided for evaluation mode")
        evaluate_model(config, args.model_path)

if __name__ == "__main__":
    main()
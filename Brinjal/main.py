import argparse
import os
from datetime import datetime
from train import train
from evaluate import evaluate
import config
from preprocessing import ImagePreprocessor
from config import PreprocessingConfig

def main():
    # --- 1. Run Preprocessing ---
    #print("--- Starting Preprocessing ---")
    #preprocessor = ImagePreprocessor(config=PreprocessingConfig)
    #preprocessor.run_pipeline()
    #print("--- Preprocessing Finished ---\n")

    # --- 2. Run Training & Evaluation ---
    parser = argparse.ArgumentParser(description='Training and evaluation pipeline.')
    parser.add_argument('--model_name', type=str, default=config.TrainingConfig.MODEL_NAME, choices=config.TrainingConfig.AVAILABLE_MODELS, help='Model to train and evaluate')
    parser.add_argument('--epochs', type=int, default=config.TrainingConfig.EPOCHS, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=config.TrainingConfig.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='models', help='Base directory under which a per-run folder is created')
    args = parser.parse_args()

    # --- Create one folder per run: models/<model_name>_<timestamp>/ ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'{args.model_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"All outputs for this run will be saved to: {run_dir}")
    print(f"{'='*80}\n")

    model_path = os.path.join(run_dir, f'{args.model_name}_final.pth')

    # Use the output directory from preprocessing as the data source for training
    training_data_dir = PreprocessingConfig.OUTPUT_DIR

    # Train the model — everything (model, json, config copy, summary, plots) lands in run_dir
    print("--- Starting Training ---")
    train(
        data_dir=training_data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_path=model_path,
        model_dir=run_dir
    )
    print("--- Training Finished ---")

    # Evaluate the model on the validation set
    print("\n--- Starting Validation Evaluation ---")
    evaluate(
        data_dir=os.path.join(training_data_dir, 'val'),
        model_name=args.model_name,
        model_path=model_path
    )
    print("--- Validation Evaluation Finished ---")

    # Evaluate the model on the test set
    print("\n--- Starting Test Evaluation ---")
    evaluate(
        data_dir=os.path.join(training_data_dir, 'test'),
        model_name=args.model_name,
        model_path=model_path
    )
    print("--- Test Evaluation Finished ---")

if __name__ == '__main__':
    main()
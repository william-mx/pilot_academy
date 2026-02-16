# Resume Training Guide

## Quick Start

### Fresh Training
```bash
python src/pilot_academy/train_pilotnet.py
```

### Resume from Checkpoint
```bash
python src/pilot_academy/train_pilotnet.py \
    resume.enabled=true \
    resume.checkpoint_path=outputs/2024-01-28/12-30-45_run/checkpoint_best.keras
```

### Resume and Train for 50 More Epochs
```bash
python src/pilot_academy/train_pilotnet.py \
    resume.enabled=true \
    resume.checkpoint_path=outputs/2024-01-28/12-30-45_run/checkpoint_best.keras \
    resume.adjust.additional_epochs=50
```

## Configuration Options

### Basic Resume
```yaml
resume:
  enabled: true
  checkpoint_path: path/to/checkpoint.keras
```

### Resume Without Optimizer State (Fresh Optimizer)
```yaml
resume:
  enabled: true
  checkpoint_path: path/to/checkpoint.keras
  restore:
    model_weights: true
    optimizer_state: false  # Start with fresh optimizer
    epoch_counter: true
```

### Resume with Reset Learning Rate
```yaml
resume:
  enabled: true
  checkpoint_path: path/to/checkpoint.keras
  adjust:
    reset_lr_schedule: true  # Restart LR schedule from beginning
```

### Resume and Train for Exact Number of Additional Epochs
```yaml
resume:
  enabled: true
  checkpoint_path: path/to/checkpoint.keras
  adjust:
    additional_epochs: 100  # Train for 100 more epochs
```

## What Gets Saved

Every checkpoint automatically saves:
1. **Model file**: `checkpoint_best.keras` or `{run_name}_final.keras`
2. **Metadata**: `checkpoint_metadata.json` containing:
   ```json
   {
     "epoch": 42,
     "best_val_loss": 0.0123,
     "final_train_loss": 0.0156,
     "total_epochs": 42,
     "config": {...}
   }
   ```

## Example Workflow

### 1. Start Training
```bash
python src/pilot_academy/train_pilotnet.py \
    train.epochs=100 \
    run.name=my_experiment
```

Training runs and saves checkpoints to:
```
outputs/2024-01-28/14-30-45_my_experiment/
├── checkpoint_best.keras
├── checkpoint_metadata.json
└── my_experiment_final.keras
```

### 2. Resume After Interruption
```bash
python src/pilot_academy/train_pilotnet.py \
    resume.enabled=true \
    resume.checkpoint_path=outputs/2024-01-28/14-30-45_my_experiment/checkpoint_best.keras \
    run.name=my_experiment_resumed
```

This will:
- Load model weights from epoch 42
- Load optimizer state
- Continue from epoch 43
- Train to original epoch 100

### 3. Continue Training Beyond Original Plan
```bash
python src/pilot_academy/train_pilotnet.py \
    resume.enabled=true \
    resume.checkpoint_path=outputs/2024-01-28/14-30-45_my_experiment/checkpoint_best.keras \
    resume.adjust.additional_epochs=50 \
    run.name=my_experiment_extended
```

This will:
- Load from epoch 42
- Train for 50 MORE epochs
- Final epoch: 92

## Advanced Usage

### Fine-tuning: Load Weights but Reset Optimizer
```bash
python src/pilot_academy/train_pilotnet.py \
    resume.enabled=true \
    resume.checkpoint_path=pretrained_model.keras \
    resume.restore.optimizer_state=false \
    resume.restore.epoch_counter=false \
    resume.adjust.reset_lr_schedule=true \
    train.optim.lr=0.00001
```

### Debug: Load Only Weights (No Epoch/Optimizer)
```bash
python src/pilot_academy/train_pilotnet.py \
    resume.enabled=true \
    resume.checkpoint_path=model_weights.keras \
    resume.restore.optimizer_state=false \
    resume.restore.epoch_counter=false
```

## Integration with WandB

Resume automatically works with WandB:

```python
# In train_pilotnet.py
wandb.init(
    project="my_project",
    name=cfg.run.name,
    config=wandb_config,
    resume="allow" if cfg.resume.enabled else False,  # ← Enables WandB resume
)
```

WandB will:
- Continue logging to the same run (if same run ID)
- Create new run with link to previous (if different run ID)
- Properly display epoch numbers

## Troubleshooting

### Error: "Checkpoint not found"
- Check the path is correct
- Use absolute path or path relative to where you run the script
- Verify the .keras file exists

### Error: "Could not load optimizer state"
- Your model architecture might have changed
- Solution: Set `resume.restore.optimizer_state=false`

### Training Starts from Epoch 0
- Check that `resume.restore.epoch_counter=true` (default)
- Verify checkpoint_metadata.json exists and has 'epoch' field

### Learning Rate Too High When Resuming
- If using LR schedule, it continues from where it left off
- Solution: Set `resume.adjust.reset_lr_schedule=true` to restart schedule

## Code Structure

### Files Involved

1. **resume_utils.py** - Core resume functionality
   - `resume_from_checkpoint()` - Load model and state
   - `calculate_epochs_for_resume()` - Calculate epoch count
   - `save_checkpoint_metadata()` - Save training metadata

2. **train_pilotnet.py** - Updated training script
   - Checks `cfg.resume.enabled`
   - Calls resume utilities
   - Adjusts training parameters

3. **config.yaml** - Main configuration
   - Contains `resume:` section
   - Can be overridden from command line

### Extending

To add custom resume behavior:

```python
# In resume_utils.py

def custom_resume_logic(model, checkpoint_info):
    """Add your custom logic here."""
    if checkpoint_info.get('best_val_loss', 1.0) < 0.01:
        print("Model was already very good, reducing learning rate")
        # Adjust learning rate or other parameters
    return model
```

## Best Practices

1. **Always save metadata**: Use `save_checkpoint_metadata()` after training
2. **Use meaningful run names**: Makes finding checkpoints easier
3. **Test resume early**: Try resuming after 1-2 epochs to verify it works
4. **Keep checkpoints**: Don't delete intermediate checkpoints until training is complete
5. **Document experiments**: Note in WandB or logs when you resume training

## Summary

Resume training is now a first-class feature with:
- ✅ Simple command-line interface
- ✅ Flexible configuration options
- ✅ Automatic metadata tracking
- ✅ WandB integration
- ✅ Safe defaults with opt-in customization

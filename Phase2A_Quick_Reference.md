# Phase 2A: Quick Reference Card
## Essential Commands and Configurations

---

## 🚀 Quick Start Commands

### 1. Check GPU
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. Load Data (Test Mode)
```python
train_dataset = SecurityDataset(
    features_path,
    labels_path,
    sequence_length=128,
    max_samples=50000  # Quick test with 50K samples
)
```

### 3. Train Model
```python
# Just run all cells in the notebook!
# Training will start automatically after data loading
```

### 4. Monitor Training
```python
# Watch for:
# - Loss decreasing
# - Accuracy increasing
# - No CUDA OOM errors
# Training curves plot automatically at the end
```

---

## ⚙️ Configuration Presets

### 🧪 Testing Configuration (Fast)
```python
CONFIG = {
    'sequence_length': 64,
    'd_model': 128,
    'nhead': 4,
    'num_encoder_layers': 3,
    'batch_size': 512,
    'num_epochs': 5,
    'learning_rate': 0.001,
}
# Use with: max_samples=10000
# Runtime: ~10 minutes
```

### 🎯 Balanced Configuration (Recommended)
```python
CONFIG = {
    'sequence_length': 128,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 6,
    'batch_size': 256,
    'num_epochs': 50,
    'learning_rate': 0.0001,
}
# Use with: Full dataset
# Runtime: ~6-8 hours
```

### 🏆 Maximum Performance Configuration
```python
CONFIG = {
    'sequence_length': 256,
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 8,
    'batch_size': 128,
    'num_epochs': 100,
    'learning_rate': 0.0001,
}
# Use with: Full dataset
# Runtime: ~15-20 hours
# Memory: Requires A100 or similar
```

---

## 🔧 Common Fixes

### CUDA Out of Memory
```python
# Quick fix:
CONFIG['batch_size'] = 64  # Reduce from 256
CONFIG['d_model'] = 128    # Reduce from 256

# Aggressive fix:
CONFIG['batch_size'] = 32
CONFIG['d_model'] = 64
CONFIG['sequence_length'] = 64
CONFIG['num_encoder_layers'] = 3
```

### Slow Training
```python
# Enable AMP (should be on by default):
CONFIG['use_amp'] = True

# Reduce logging:
CONFIG['log_interval'] = 500  # from 100

# Increase batch size (if memory allows):
CONFIG['batch_size'] = 512
```

### Model Not Learning
```python
# Increase learning rate:
CONFIG['learning_rate'] = 0.001  # from 0.0001

# Increase warmup:
CONFIG['warmup_steps'] = 2000  # from 1000

# Check data:
# - Verify labels are correct
# - Check for NaN values
# - Ensure data is normalized
```

### Overfitting
```python
# Increase regularization:
CONFIG['dropout'] = 0.3  # from 0.1
CONFIG['weight_decay'] = 0.05  # from 0.01

# Reduce model size:
CONFIG['num_encoder_layers'] = 4  # from 6
CONFIG['d_model'] = 128  # from 256

# Use early stopping (already enabled):
CONFIG['early_stopping_patience'] = 5  # from 7
```

---

## 📊 Interpreting Metrics

### Training Metrics (Console Output)
```
Epoch 10/50
──────────────────────────────────────
Train Loss: 0.0234 | Train Acc: 99.12%
Val Loss: 0.0289 | Val Acc: 98.87%
Val Precision: 0.9891 | Val Recall: 0.9884
Val F1: 0.9887 | Val FPR: 0.0234

✓ Best model saved  # ← Good! Model improving
```

### What's Good?
✅ Loss: <0.05
✅ Accuracy: >98%
✅ F1 Score: >0.95
✅ FPR: <0.05 (5%)

### What's Bad?
❌ Loss: >0.2 or increasing
❌ Accuracy: <90%
❌ F1 Score: <0.80
❌ FPR: >0.15 (15%)

---

## 🎯 Expected Results

### Your Research Targets
- **Accuracy**: >99% (match baseline)
- **F1 Score**: >0.98
- **False Positive Rate**: <5%
- **Inference Time**: <10ms per sequence

### Realistic First Run Results
- **Accuracy**: 96-98%
- **F1 Score**: 0.94-0.96
- **FPR**: 5-8%
- **Training Time**: 6-8 hours (full dataset)

### After Hyperparameter Tuning
- **Accuracy**: 98-99.5%
- **F1 Score**: 0.97-0.99
- **FPR**: 2-5%

---

## 💾 Important Files

### Outputs Created
```
checkpoints/
├── best_model.pth                    # Best validation model
├── transformer_component_final.pth   # For Phase 2C
├── checkpoint_epoch_X.pth            # Per-epoch saves
├── training_curves.png               # Loss/accuracy plots
├── confusion_matrix.png              # Confusion matrix
└── per_class_metrics.png             # Per-class performance
```

### What to Keep for Research
- ✅ `best_model.pth` - For evaluation
- ✅ `transformer_component_final.pth` - For ensemble
- ✅ All PNG files - For dissertation
- ❌ `checkpoint_epoch_X.pth` - Can delete after training

---

## 🔍 Monitoring During Training

### What to Watch
1. **GPU Memory Usage**
   - Should be ~80-90% utilized
   - If <50%: Increase batch size
   - If >95%: Reduce batch size

2. **Training Speed**
   - ~50-100 batches/second (T4 GPU)
   - If slower: Reduce num_workers or disable AMP

3. **Loss Curve**
   - Should decrease smoothly
   - Small fluctuations OK
   - Large spikes = problem

4. **Validation Metrics**
   - Should improve over time
   - If plateauing: Continue training
   - If degrading: Overfitting

---

## 🐛 Debug Checklist

### Before Training
- [ ] GPU is enabled and detected
- [ ] Data files are accessible
- [ ] Data loads successfully (test 1 batch)
- [ ] Model forward pass works
- [ ] Batch size fits in memory

### During Training
- [ ] Loss is decreasing
- [ ] Accuracy is increasing
- [ ] No CUDA errors
- [ ] GPU memory stable
- [ ] Checkpoints being saved

### After Training
- [ ] Training curves look reasonable
- [ ] Best model saved successfully
- [ ] Test metrics calculated
- [ ] Confusion matrix generated
- [ ] Results documented

---

## 📞 Emergency Debugging

### Notebook Crashes
```python
# Restart runtime:
# Runtime → Restart runtime

# Clear GPU memory:
torch.cuda.empty_cache()

# Reduce everything:
CONFIG['batch_size'] = 32
CONFIG['d_model'] = 64
CONFIG['num_encoder_layers'] = 2
```

### Data Loading Errors
```python
# Check paths:
import os
print(os.listdir(CONFIG['data_dir']))

# Test load one file:
df = pd.read_csv('path/to/file.csv', nrows=10)
print(df.head())
```

### Can't Load Trained Model
```python
# Load checkpoint manually:
checkpoint = torch.load('checkpoints/best_model.pth', 
                       map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🎓 For Dissertation

### Key Numbers to Report
```python
# Model Architecture
Total Parameters: ~6.5M
Encoder Layers: 6
Attention Heads: 8
Embedding Dimension: 256

# Training
Dataset: CICIDS-2017 (CTGAN augmented)
Training Samples: ~500K
Validation Samples: ~100K
Test Samples: ~100K
Training Time: ~6-8 hours (T4 GPU)

# Performance
Accuracy: [YOUR RESULT]%
Precision: [YOUR RESULT]
Recall: [YOUR RESULT]
F1 Score: [YOUR RESULT]
False Positive Rate: [YOUR RESULT]%
```

### Figures to Include
1. Training curves (loss and accuracy)
2. Confusion matrix
3. Per-class metrics bar chart
4. Architecture diagram (from guide)

---

## ⏭️ After Phase 2A

### Phase 2B Preparation
```python
# Save transformer outputs for ensemble:
# (Code provided in Phase 2B notebook)

# Document performance:
# - Accuracy, F1, FPR
# - Training time
# - Inference speed
# - Memory usage

# Prepare for GNN:
# - Graph structure design
# - Node/edge features
# - Similar training pipeline
```

---

## 📊 Experiment Tracking Template

```markdown
### Experiment: [Name/Date]

**Configuration:**
- Sequence Length: 128
- Model Size: 256
- Batch Size: 256
- Learning Rate: 0.0001
- Epochs: 50

**Results:**
- Train Acc: ___%
- Val Acc: ___%
- Test Acc: ___%
- F1 Score: ____
- FPR: ____%
- Training Time: ___ hours

**Observations:**
- [What worked well]
- [What didn't work]
- [Ideas for improvement]

**Next Steps:**
- [ ] Try [modification 1]
- [ ] Test [configuration 2]
- [ ] Analyze [specific issue]
```

---

## 💡 Pro Tips

1. **Start small**: Test with 10K samples first
2. **Save often**: Checkpoints every epoch
3. **Monitor GPU**: Use `nvidia-smi` command
4. **Document everything**: Keep experiment log
5. **Compare carefully**: Use same metrics as baseline
6. **Visualize results**: Plots help understand performance
7. **Test inference**: Verify real-time capability

---

## 🆘 Common Error Messages

### "RuntimeError: CUDA out of memory"
→ Reduce batch_size or model size

### "ValueError: empty range for randrange"
→ Check dataset has enough samples

### "FileNotFoundError: [file] not found"
→ Verify data_dir path is correct

### "RuntimeError: Expected all tensors on same device"
→ Ensure model and data both on GPU

### "KeyError: 'model_state_dict'"
→ Checkpoint file corrupted, use backup

---

**Keep this reference handy while implementing! 📌**

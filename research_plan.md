# Kế Hoạch Nghiên Cứu Luận Văn Thạc Sĩ
## Áp Dụng DD3D cho Phát Hiện Máy Bay 3D ở Khoảng Cách Xa

### 1. DATASET ANALYSIS & PREPARATION

#### 1.1 Phân tích đặc điểm dữ liệu aviation
```python
# Phân tích distribution
- Khoảng cách: >350m (vs KITTI ~50m)
- Kích thước object: rất nhỏ trong ảnh
- Background: sky, clouds (vs road scenes)
- Lighting conditions: variable sky conditions
- Camera setup: ground-based looking up
```

#### 1.2 Dataset preprocessing pipeline
- Atmospheric correction
- Scale normalization
- Augmentation strategies cho small objects
- 3D annotation quality validation

### 2. BASELINE EXPERIMENTS

#### 2.1 DD3D Original Performance
```bash
# Experiment 1: DD3D trên aviation data
./scripts/train.py +experiments=dd3d_aviation_dla34_baseline
```

#### 2.2 Performance Analysis
- mAP@different distances (350m, 500m, 1km+)
- Scale-wise performance breakdown
- Failure case analysis

### 3. PROPOSED IMPROVEMENTS

#### 3.1 Scale-Aware Feature Pyramid
- Multi-scale feature extraction
- Adaptive receptive field
- Small object enhancement

#### 3.2 Long-Range Depth Estimation
- Atmospheric model integration
- Multi-frame depth consistency
- Uncertainty-aware depth prediction

#### 3.3 Domain Adaptation
- Progressive domain transfer
- Adversarial training
- Synthetic-to-real adaptation

### 4. EXPERIMENTAL VALIDATION

#### 4.1 Ablation Studies
- Each component contribution
- Hyperparameter sensitivity
- Architecture choices

#### 4.2 Comparison Studies
- vs Original DD3D
- vs FCOS3D, SMOKE, MonoDLE
- vs Traditional aviation detection

### 5. EVALUATION METRICS

#### Aviation-specific metrics:
- Distance-stratified mAP
- Angular accuracy
- Altitude estimation error
- False positive analysis in sky regions

### 6. TIMELINE (6 tháng)

**Tháng 1-2: Literature Review + Dataset**
- State-of-art survey
- Dataset preparation
- Baseline implementation

**Tháng 3-4: Method Development**
- Scale-aware improvements
- Long-range depth estimation
- Domain adaptation

**Tháng 5: Experiments & Evaluation**
- Comprehensive experiments
- Ablation studies
- Comparison studies

**Tháng 6: Thesis Writing**
- Results analysis
- Discussion & conclusion
- Thesis finalization

### 7. EXPECTED CONTRIBUTIONS

1. **Novel Aviation 3D Dataset** with long-range annotations
2. **Scale-Aware DD3D** for small object detection
3. **Long-Range Depth Estimation** techniques
4. **Domain Adaptation** framework aviation←→automotive
5. **Comprehensive Evaluation** framework for aviation 3D detection

### 8. POTENTIAL PUBLICATIONS

1. **Conference paper**: Scale-Aware DD3D for Long-Range Aircraft Detection
2. **Workshop paper**: Domain Adaptation for 3D Object Detection
3. **Dataset paper**: Aviation 3D Detection Benchmark
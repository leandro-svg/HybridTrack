# HybridTrack: A Hybrid Approach for Robust Multi-Object Tracking

[📄 Read the paper on arXiv](https://www.arxiv.org/abs/2501.01275)

## 🏆 News

- HybridTrack has been **accepted to RAL 2025**!

---
HybridTrack is a novel 3D multi-object tracking (MOT) framework that combines the strengths of traditional Kalman filtering with the adaptability of deep learning. Designed for traffic and autonomous driving scenarios, it delivers **state-of-the-art accuracy** and **real-time performance**—no manual tuning or scenario-specific designs required.

---

## 🚀 Highlights

### 📦 Features
- 3D Object Tracking using LiDAR
- Learnable Kalman Filter (LKF)
- Real-time performance (112 FPS)
- High tracking accuracy without handcrafted noise or motion models
- Generalizes across different driving scenarios

### 📊 Benchmark Performance
| Method               | HOTA     | FPS   | Modality   | Model Weights                                                                 |
|----------------------|----------|-------|------------|-------------------------------------------------------------------------------|
| HybridTrack (Ours)   | **82.08%** | **112** | 3D (LiDAR) | [Download (.pth)](https://drive.google.com/file/d/1beFjycNjTtb2nDDf0vteHp1NNbR4lrvR/view?usp=sharing) |
| PMTrack              | 81.36%   | -     | 3D         | -                                                                             |
| PC-TCNN              | 80.90%   | -     | 3D         | -                                                                             |
| UG3DMOT              | 78.60%   | -     | 3D         | -                                                                             |

See the paper for detailed comparison across metrics like MOTA, IDF1, and association accuracy.

### 📁 Dataset
HybridTrack is evaluated on the [KITTI Tracking Benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php).

---

## ⚡ Quickstart

1. **Prepare your data:** Follow the [Data Preparation Guide](docs/create_data.md) for step-by-step instructions on downloading, organizing, and linking the KITTI dataset, detections, and annotations.
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure and run:**
   - For training, see [Training Guide](docs/training.md)
   - For tracking, see [Tracking Guide](docs/tracking.md)

---

## 📚 Documentation
- [Data Preparation Guide](docs/create_data.md)
- [Training Guide](docs/training.md)
- [Tracking Guide](docs/tracking.md)

---

## 🖼️ Example Results
![Tracking Results](hybridtrack_demo.gif)

---
## 🏗️ Method Architecture

> 📌 *A schematic of the HybridTrack architecture will be added here.*

![Method Architecture](model_architecture_11.jpg)

---

## 📝 Project Status

🚧 **Code will be released upon acceptance. Stay tuned!**

---



## 📜 Citation
If you use HybridTrack in your research, please consider citing:

```bibtex
@misc{dibella2025hybridtrackhybridapproachrobust,
      title={HybridTrack: A Hybrid Approach for Robust Multi-Object Tracking}, 
      author={Leandro Di Bella and Yangxintong Lyu and Bruno Cornelis and Adrian Munteanu},
      year={2025},
      eprint={2501.01275},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01275}, 
}
```

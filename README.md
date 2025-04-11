# HybridTrack: A Hybrid Approach for Robust Multi-Object Tracking

[📄 Read the paper on arXiv](https://www.arxiv.org/abs/2501.01275)

HybridTrack is a novel 3D multi-object tracking (MOT) framework that integrates the strengths of traditional Kalman filtering with the adaptability of deep learning. Designed for use in traffic and autonomous driving scenarios, it delivers **state-of-the-art accuracy** and **real-time performance**, eliminating the need for manual tuning or scenario-specific designs.

## 🔧 Project Status

🚧 **Code will be released upon acceptance. Stay tuned!**

---

## 🔍 Overview

HybridTrack introduces a **learnable Kalman filter** that dynamically adjusts motion and noise parameters using a lightweight deep learning architecture. It operates under a tracking-by-detection (TBD) paradigm and has been validated on the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/), achieving:

- **82.08% HOTA** – Higher Order Tracking Accuracy
- **112 FPS** – Real-time tracking speed
- **Superior generalization** to various scenarios without agent-specific modeling

---

## 📐 Method Architecture

> 📌 *A schematic of the HybridTrack architecture will be added here.*

![Method Architecture Placeholder](./assets/hybridtrack_architecture.png)

---

## 🎞️ Example Results

> 📌 *Example visualizations and qualitative tracking results will be provided here.*

![Tracking Results Placeholder](./assets/hybridtrack_demo.gif)

---

## 📦 Features

- 3D Object Tracking using LiDAR
- Learnable Kalman Filter (LKF)
- Real-time performance (112 FPS)
- High tracking accuracy without handcrafted noise or motion models
- Generalizes across different driving scenarios

---

## 📊 Benchmark Performance

| Method         | HOTA | FPS  | Modality |
|----------------|------|------|----------|
| HybridTrack (Ours) | **82.08%** | **112** | 3D (LiDAR) |
| PMTrack        | 81.36% | -    | 3D       |
| PC-TCNN        | 80.90% | -    | 3D       |
| UG3DMOT        | 78.60% | -    | 3D       |

📌 See the paper for detailed comparison across metrics like MOTA, IDF1, and association accuracy.

---

## 📁 Dataset

HybridTrack is evaluated on the [KITTI Tracking Benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php). Please follow their instructions to download the dataset and annotations.

---

## 📅 TODO

- [ ] Release source code & training scripts
- [ ] Upload pretrained models
- [ ] Add detailed documentation and setup instructions
- [ ] Add demo video and inference guide

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

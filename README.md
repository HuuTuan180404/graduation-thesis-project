# Graduation thesis

## 1. Overview

This project addresses the problem of automatic Sign Language Recognition (SLR), which aims to recognize and classify sign gestures from video sequences. Sign language recognition plays a crucial role in facilitating communication between the Deaf community and hearing individuals by enabling intelligent translation and interaction systems.

In this work, we focus on isolated sign recognition using visual inputs, particularly skeleton-based representations extracted from video data. The proposed approach is designed to effectively model spatial-temporal patterns of human body and hand movements.

Extensive experiments are conducted on two widely used benchmark datasets:

- **WLASL100** – A subset of the Word-Level American Sign Language (WLASL) dataset containing 100 sign classes.
- **LSA64** – An Argentine Sign Language dataset consisting of 64 isolated sign categories.

Experimental results demonstrate the effectiveness of the proposed model compared to existing state-of-the-art methods on both datasets.

## 2. System Architecture Diagram

<p align="center">
  <img src="images/architecture.svg" width="80%" style="border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.12);">
  <br>
  <figcaption align="center">
    <em>Figure 1. Proposed architecture.</em>
  </figcaption>
</p>

## 3. Datasets

### 4.1 WLASL100

**WLASL100** is a subset of the Word-Level American Sign Language (WLASL) dataset, which is one of the largest publicly available benchmarks for isolated sign language recognition. The full WLASL dataset contains over 2,000 sign categories collected from real-world video sources.

In this work, we use the **WLASL100** subset, which includes:

- 100 isolated sign classes
- 2038 videos
- 97 signers

WLASL100 is considered a challenging benchmark due to large intra-class variations and signer diversity, making it suitable for evaluating the robustness and generalization ability of sign language recognition models.

---

### 4.2 LSA64

**LSA64** is a benchmark dataset for isolated sign recognition in Argentine Sign Language (LSA). It consists of:

- 64 isolated sign classes
- 3200 videos
- 10 signers

Compared to WLASL100, LSA64 is more constrained in terms of background and recording setup, which allows for evaluating model performance under relatively cleaner conditions.

---

Both datasets are widely used in the literature for benchmarking SOTA sign language recognition methods.

### 4.3 Data Processing

The data preprocessing pipeline follows the procedure described in the [Siformer paper](https://dl.acm.org/doi/10.1145/3664647.3681578).

### 4.4 Data Flow

<p align="center">
  <img src="images/data-flow.svg" width="80%" style="border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.12);">
  <br>
  <figcaption align="center">
    <em>Figure 2. Data flow diagram.</em>
  </figcaption>
</p>

## 4. SOTA

<div align="center">

### 📊 Comparison of SOTA Methods on the WLASL100 Dataset

#### 🔹 Skeleton-Based Methods

| Method             | Top-1 Acc (%) | Top-5 Acc (%) |
| ------------------ | :-----------: | :-----------: |
| ST-GCN             |     50.78     |     79.07     |
| Pose-TGCN          |     55.43     |     78.68     |
| SPOTER             |     63.18     |      --       |
| SignBERT+          |     79.84     |     91.09     |
| SIGNGRAPH          |     72.09     |     88.76     |
| Siformer           |     86.50     |      --       |
| **Proposed Model** |   **87.5**    |   **93.75**   |

---

#### 🔹 Other Data-Based Methods

| Method    | Top-1 Acc (%) | Top-5 Acc (%) |
| --------- | :-----------: | :-----------: |
| I3D       |     65.89     |     84.11     |
| TCK       |     77.52     |     91.08     |
| SignBERT+ |     84.11     |     96.51     |
| Fusion-3  |     75.67     |     86.00     |
| Uni-Sign  |   **92.25**   |      --       |
| NLA-SLR   |     91.47     |   **96.90**   |

---

**Table 1.** Performance comparison on the WLASL100 dataset.

</div>

<div align="center">

### 📊 Comparison of State-of-the-Art Methods on the LSA64 Dataset

| Method             | Accuracy (%) |
| ------------------ | :----------: |
| MEMP               |    99.06     |
| I3D                |    98.91     |
| SPOTER             |   **100**    |
| SIGNGRAPH          |   **100**    |
| Siformer           |    99.84     |
| **Proposed Model** |   **100**    |

---

**Table 2.** Performance comparison of SOTA methods on the LSA64 dataset.

</div>

<!-- ## 7. Limitations -->

<!-- ## 8. Future Improvements -->

## 5. Source Code

The source code of this project is organized into different branches corresponding to each experimental dataset.

To reproduce the experiments, please switch to the appropriate branch:

- **WLASL100 experiments**

  ```bash
  git checkout wlasl100
  ```

- **LSA64 experiments**

  ```bash
  git checkout lsa64
  ```

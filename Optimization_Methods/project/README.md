# Project: Stochastic Enhancement of Graph and Rank Regularized Matrix Recovery (GRMR) for Snapshot Spectral Image Demosaicing

**Collaborators:** Developed by me and [Nikos Kontogeorgis](https://github.com/NikosKont)

**Description:**

This project focuses on advancing the methodology of the Graph and Rank Regularized Matrix Recovery (GRMR) algorithm, initially proposed for Snapshot Spectral Image Demosaicing. Snapshot Spectral Imaging (SSI) offers an efficient solution for capturing spatio-spectral information in dynamic scenes on compact platforms, yet faces a crucial trade-off between spatial and spectral resolution. To address this, we introduce a stochastic modification to the GRMR algorithm, enabling enhanced rank flexibility and improved image fidelity.

**Key Contributions:**

- **Stochastic Rank Selection**: A novel stochastic element is introduced in the GRMR algorithm's rank determination process, leveraging randomness to improve matrix recovery outcomes by dynamically adjusting rank in each iteration.
- **Improved Reconstruction Fidelity**: Our empirical tests on the same datasets as the original GRMR study demonstrate that this stochastic approach yields higher Peak Signal-to-Noise Ratio (PSNR) and lower Spectral Angle Mapper (SAM) scores, indicating superior spectral and spatial accuracy.
- **Real-World Applicability**: The proposed enhancement to GRMR was validated on standard datasets such as CAVE and Sentinel-2, showing promising results that suggest this approach could be valuable for applications in precision agriculture, climate monitoring, and more.

The project provides:

- **Theoretical Foundations**: A deep dive into the GRMR framework and how spectral imaging can be optimized through graph and rank regularization.
- **Algorithmic Enhancements**: Detailed steps and modifications to the GRMR algorithm, introducing a stochastic approach that uses random rank selection in matrix recovery.
- **Comparative Analysis**: Performance comparisons between GRMR and sGRMR on multiple metrics, demonstrating the improved reconstruction fidelity with the stochastic approach.

For more details, refer to the full project report: [Project PDF](report.pdf).

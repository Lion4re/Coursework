# CS573 - Optimization Methods Assignments

This repository contains assignments from the course CS573 - Optimization Methods, offered by the Department of Computer Science at the University of Crete during the Fall Semester 2023-2024.

## Course Description

This repository contains assignments and related materials for the CS573 - Optimization Methods course. The course covers various optimization techniques and algorithms used in computer science and engineering.

## Assignments

### Assignment 1: Singular Value Decomposition and Principal Component Analysis

**Description:**

Assignment 1 focuses on applying Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) to analyze datasets. The assignment includes:

- Loading and preprocessing datasets.
- Implementing SVD to decompose matrices.
- Utilizing PCA for dimensionality reduction.
- Analyzing principal components and their significance.
- Conducting nearest neighbor analysis in low-dimensional space.
- Visualizing data distributions and patterns.

For more details, refer to the [Assignment 1 PDF](assignment1/HY573_ex1.pdf).

### Assignment 2: Regression with Missing Data

**Description:**

Assignment 2 provides a comprehensive study of regression analysis, starting with Least Squares Regression and extending to Lasso Regression with a focus on feature selection and regularization. It further explores the concept of matrix completion as a method to handle missing data in datasets.

The assignment includes:

- Implementing Least Squares Regression from scratch to understand the fundamentals of linear modeling.
- Developing a Lasso Regression model to incorporate L1 regularization for feature selection and model simplicity.
- Investigating the impact of missing data on model performance and exploring matrix completion as a solution.
- Re-evaluating the regression models after applying matrix completion to see the improvements in prediction accuracy.

For more details, refer to the [Assignment 2 PDF](assignment2/HY573_ex2.pdf).

### Assignment 2: Regression with Missing Data

**Description:**

Assignment 2 provides a comprehensive study of regression analysis, starting with Least Squares Regression and extending to Lasso Regression with a focus on feature selection and regularization. It further explores the concept of matrix completion as a method to handle missing data in datasets.

The assignment includes:

- Implementing Least Squares Regression from scratch to understand the fundamentals of linear modeling.
- Developing a Lasso Regression model to incorporate L1 regularization for feature selection and model simplicity.
- Investigating the impact of missing data on model performance and exploring matrix completion as a solution.
- Re-evaluating the regression models after applying matrix completion to see the improvements in prediction accuracy.

For more details, refer to the [Assignment 2 PDF](assignment2/HY573_ex2.pdf).

### Project: Stochastic Enhancement of Graph and Rank Regularized Matrix Recovery (GRMR) for Snapshot Spectral Image Demosaicing

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

For more details, refer to the full project report: [Project PDF](project/report.pdf).


**Report:**

Each assignment in this repository includes a detailed report. The reports provide insights, analyses, and discussions on the implemented methods, results, and the practical implications of optimization techniques in solving real-world problems.

## Folder Structure

- `Assignment1/`: Contains the code and materials for Assignment 1.
- `Assignment2/`: Contains the code and materials for Assignment 2.
- `project/`: Contains the report and the presentation of the course project.


## Tools and Languages

- **Python**: Main programming language used for model development and data analysis.
- **Jupyter Notebooks**: For interactive code execution and visualization.
- **NumPy & Pandas**: Libraries used for numerical computations and data manipulation.
- **Matplotlib & Seaborn**: For data visualization.

## Viewing the Assignments

To view the assignments, download the respective PDF files and open them with any standard PDF viewer.

## Running the Code

The code for the assignments can be found in Jupyter Notebooks. To run the notebooks:

1. Ensure you have Jupyter installed. If not, install it using `pip install jupyter`.
2. Navigate to the directory containing the `.ipynb` files.
3. Run `jupyter notebook` to launch the Jupyter Notebook App or your preffered editor/IDE.

## Acknowledgements

- University of Crete, Department of Computer Science
- Course Instructor: [Tsagkatakis Grigorios]([Instructor's Profile Link](https://users.ics.forth.gr/~greg/)https://users.ics.forth.gr/~greg/)

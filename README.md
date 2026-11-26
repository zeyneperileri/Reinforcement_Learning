#  Reinforcement Learning: Mathematical Foundations & Algorithms

Welcome to my Reinforcement Learning repository. This space serves as a structured, progressive portfolio documenting my independent study and algorithmic implementations in the field of Reinforcement Learning.

## 📖 Core Reference
The theoretical backbone and mathematical derivations within this repository are strictly aligned with my ongoing study of:
> **Mathematical Foundations of Reinforcement Learning** by *Shiyu Zhao*.

As I progress through the chapters, my goal is to translate the core theorems and equations into computationally efficient Python algorithms.

## 📍 Table of Contents
This repository is structured sequentially to reflect the learning progression of RL concepts. Each directory contains the source code and a detailed explanation of the mathematical approach used.

*   [**Shared Environment: Grid World**](./env)
    *   *Reusable 2D Grid World supporting both model-based (DP) and model-free (MC/TD) RL algorithms with Matplotlib visualization.*
*   [**01. Iterative Policy Evaluation**](./01_Iterative_Policy_Evaluation) 
    *   *Solution of the General Bellman Equation using iterative state-value updates.*
*   [**02. Bellman Optimality Equation (BOE)**](./02_Bellman_Optimality_Equation)
    *   *General matrix-vector Value Iteration solver based on the Contraction Mapping Theorem.*
*   [**03. Optimal Policy Analysis**](./03_Optimal_Policy_Analysis)
    *   *Sensitivity analysis of optimal policies under varying penalties, discount rates ($\gamma$), and affine reward transformations using the shared Grid World.*
*   *(More chapters and algorithms will be added progressively...)*

## 🛠️ Technologies & Tools
*   **Python 3**
*   **NumPy** (Utilized for vectorized matrix operations and norm calculations)
*   **Matplotlib** (Used for visualizing Grid World layouts, state values, and optimal policies)

---
*Maintained by Zeynep Erileri*

# Training Program 2025

This repository consolidates the demos, models, evaluation pipelines, datasets, and development environments used across the **Training Program 2025**. The structure reflects different stages of the workflow, from environment setup and model development to evaluation and external code integration.

---

## Repository Structure

### 1. `demo_1/` — Environment Checker

Interactive setup-validation module built with **Python** and **Streamlit**.

Includes:
- A local interactive web interface
- Guided environment setup checks
- System health verification
- WSL and GPU driver checks
- Conda environment verification
- Docker functionality testing
- Core Python library import checks

This module helps users confirm that their machine and software environment are correctly configured before starting the rest of the program.

---

### 2. `demo_2/` — Model Development

Primary workspace for building, training, and iterating on the core model.

Includes:
- Main model implementation
- `.devcontainer` for reproducible development
- Training and testing scripts
- Supporting resources for experimentation

This module functions as the foundation for downstream evaluation workflows.

---

### 3. `demo_3/` — Forecast Evaluation and Model Comparison

Interactive evaluation and benchmarking module developed as a **Google Colab / web-based experience** in collaboration with **Rhiza**, presented through the **Genevieve website**.

Includes:
- Forecast verification workflows
- Interactive scorecard generation
- Comparative visualizations and summary tables
- Benchmarking across multiple weather models
- Exploration of forecast verification metrics such as RMSE, ACC, CRPS, and SEEPS
- Ground-truth dataset selection for evaluation analysis

This module helps participants understand how forecasting systems are professionally evaluated and compared.

---

### 4. `demo_4_5/` — Evaluation and International Data

Module for running evaluation pipelines and analyzing cross-country datasets.

Includes:
- Evaluation notebooks and scripts aligned with `demo_2`
- Data contributions from participating countries
- `.devcontainer` for consistent execution environments
- Metrics, reporting, and model comparison utilities

This module provides the benchmarking and analytics capabilities for the program.

---

### 5. `g42/` — External Model Code

Dedicated environment for integrating and testing code originating from G42.

Includes:
- External model components
- Experimental algorithms or reference implementations
- `.devcontainer` for isolated and reproducible execution

This module serves as a comparative baseline and complementary model workspace.

---

## Dev Containers

Each major development module (`demo_2`, `demo_4_5`, `g42`) includes its own `.devcontainer` to ensure:
- Consistent and reproducible development environments
- Dependency isolation
- Turnkey setup through VS Code DevContainers

---

## Purpose

The repository is designed to centralize:
- Environment setup and validation
- Model development workflows
- Forecast evaluation and benchmarking
- Shared datasets
- External model integrations
- Standardized development environments

---

## Getting Started

1. Start with `demo_1` to verify that your local environment is correctly configured.
2. Use `demo_2` for model development and experimentation.
3. Explore `demo_3` to understand forecast verification and model comparison.
4. Run evaluation workflows in `demo_4_5`.
5. Review `g42` for external model integrations and comparative implementations.

---

## Notes

Each demo or module may include its own local documentation and instructions.  
Please check the corresponding folder `README.md` for more detailed guidance.

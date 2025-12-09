Training Program 2025
This repository consolidates the models, evaluation pipelines, datasets, and development environments used across the Training Program 2025. The structure reflects the full workflow lifecycle: model development, evaluation, and external code integration.

Repository Structure
1. demo_2/ — Model Development
Primary workspace for building, training, and iterating on the core model.

Includes: - Main model implementation - .devcontainer for reproducible development - Training and testing scripts - Supporting resources for experimentation

This module functions as the foundation for downstream evaluation workflows.

2. demo_4_5/ — Evaluation and International Data
Module for running evaluation pipelines and analyzing cross-country datasets.

Includes: - Evaluation notebooks and scripts aligned with demo_2 - Data contributions from participating countries - .devcontainer for consistent execution environments - Metrics, reporting, and model comparison utilities

This module provides the benchmarking and analytics capabilities for the program.

3. g42/ — External Model Code
Dedicated environment for integrating and testing code originating from G42.

Includes: - External model components - Experimental algorithms or reference implementations - .devcontainer for isolated and reproducible execution

This module serves as a comparative baseline and complementary model workspace.

Dev Containers
Each major module (demo_2, demo_4_5, g42) includes its own .devcontainer to ensure: - Consistent and reproducible development environments - Dependency isolation - Turnkey setup through VS Code DevContainers

Purpose
The repository is designed to centralize: - Model development workflows - Evaluation pipelines - Shared datasets - External model integrations - Standardized development environments

Getting Started
Navigate to the module you plan to work with (demo_2, demo_4_5, or g42).
Open the module inside a DevContainer-enabled environment (VS Code recommended).
Review and follow any module-specific guidance.
Execute training, evaluation, or exploratory workflows as required.

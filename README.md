📘 Training Program 2025 – Repository

This repository contains the materials, models, evaluation pipelines, and supporting code used throughout the Training Program 2025.
It is organized into three main modules, each representing a different stage of the workflow: model development, evaluation, and external code integration.

📂 Repository Structure
1️⃣ demo_2/ – Model Development

This folder contains:

The main model implementation developed during the training program

A .devcontainer setup for reproducible development environments

Supporting scripts and resources used to train and test the model

This is the foundational module that the rest of the project builds on.

2️⃣ demo_4_5/ – Evaluation & International Data

This module includes:

Evaluation pipelines and notebooks related to demo_2

Data shared by participating countries

A .devcontainer folder for consistent environment setup

Scripts used for metrics, reporting, and model comparisons

This folder represents the analysis and evaluation stage of the training program.

3️⃣ g42/ – External Model Code

This folder contains:

Code originating from G42

Likely includes model components, experiments, or algorithmic contributions

A .devcontainer environment for running the code consistently

This module may serve as a reference implementation or additional model used during the program.

🛠️ Dev Containers

Each major folder (demo_2, demo_4_5, g42) includes its own .devcontainer directory, providing:

Consistent development environments

Dependency isolation

Ready-to-use VS Code DevContainer configurations

This ensures that all contributors can run and test the code in identical environments.

🎯 Purpose of This Repository

The goal of this repository is to centralize:

Model implementations

Evaluation workflows

Shared datasets

External reference code

Reproducible development environments

It supports training, experimentation, and collaboration across teams.

🔧 How to Use This Repository

Navigate to the module you want to work on (demo_2, demo_4_5, or g42)

Open it in a DevContainer-enabled environment (VS Code recommended)

Follow the module’s internal instructions (if any)

Run training, evaluation, or exploration scripts as needed



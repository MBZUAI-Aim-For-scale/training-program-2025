# Demo 3 - Forecast Evaluation and Model Comparison

## Overview

**Demo 3** focuses on **evaluating and comparing weather forecasting models** instead of simply running them.  
It gives participants a practical and intuitive understanding of how forecasting models are **benchmarked professionally** using different verification metrics.

This demo was developed as a **interactive online platform** in collaboration with **Rhiza**, and it is presented through the **Genevieve website**. The platform allows users to explore forecast performance through an interactive interface, generate scorecards, and analyze visual results across different models, variables, regions, and datasets.

---

## Purpose

The purpose of this demo is to help users understand **how forecast quality is measured** and how model performance can vary depending on the chosen metric, variable, forecast lead time, and ground-truth dataset.

Instead of only producing forecasts, this demo teaches users how to **interpret verification results** and compare the strengths and weaknesses of different AI and operational weather models.

---

## What This Demo Does

The demo provides an **interactive web-based evaluation tool** where users can configure a forecast comparison and instantly visualize the results.

It allows participants to:

- choose a verification metric
- compare multiple forecasting models
- select meteorological variables and regions
- choose the appropriate ground-truth dataset
- generate visual scorecards and summary tables
- answer guided interpretation questions

This makes the learning process more hands-on and exploratory.

---

## Learning Objectives

By completing this demo, participants should be able to:

- define and distinguish between key forecast verification metrics:
  - **RMSE** (Root Mean Square Error)
  - **ACC** (Anomaly Correlation Coefficient)
  - **CRPS** (Continuous Ranked Probability Score)
  - **SEEPS** (Stable Equitable Error in Probability Space)

- interpret metric outputs across:
  - different variables
  - different forecast lead times
  - different models

- understand the strengths, weaknesses, and use cases of each metric

- explain why different variables may require different evaluation methods

- recognize how the choice of **ground-truth dataset** can influence evaluation results

- generate and analyze plots and tables to compare AI models with operational forecasting systems

---

## Platform and Medium

- **Medium:** Interactive online platform / website
- **Implementation context:** Google Colab with **Rhiza**
- **Presentation platform:** **Genevieve website**

---

## Process

### 1. Introduction and Orientation
The user opens the web page and is introduced to the concept of **forecast verification** and the purpose of the tool.

This section explains that the platform is designed to help users compare model performance using recognized meteorological evaluation metrics.

### 2. Build Your Comparison
The user interacts with a clean control panel that contains dropdown menus, checkboxes, and guided options to configure the analysis.

#### A. Select a Metric
The user selects a primary forecast verification metric from a dropdown menu, such as:

- **RMSE**
- **ACC**
- **CRPS**
- **SEEPS**

#### B. Contextual Helper Text
Once a metric is selected, the interface displays a **"Learn About This Metric"** box.

For example, for **RMSE**, the tool explains:

- **What it is:** Root Mean Square Error, which measures the average magnitude of forecast errors in the same units as the variable
- **How to interpret it:** Lower values are better, and a value of 0 indicates a perfect forecast
- **Strengths and caveats:** RMSE is easy to understand, but it strongly penalizes large errors and does not describe the nature of the error, such as systematic bias

This contextual explanation helps users learn the meaning of each metric while using the tool.

#### C. Select Models
The user selects the models they want to compare using checkboxes.

Examples include:

- **AIFS**
- **GraphCast**
- **FuXi**
- **IFS-HRES**

#### D. Select Variable and Region
The user chooses:

- a **meteorological variable**, such as:
  - Geopotential at 500 hPa
  - 2-meter Temperature
  - Total Precipitation

- a **geographical region**, such as:
  - Global
  - Europe
  - North America

#### E. Select Ground Truth
The user then selects the **ground-truth verification dataset**.

The tool intelligently filters the available options depending on the selected variable.

For example:

- if **Total Precipitation** is selected, the user may choose from:
  - **ERA5**
  - **IMERG**
  - **CHIRPS**

- if **Geopotential** or **Temperature** is selected, the available ground-truth dataset may only be:
  - **ERA5**

This step highlights that evaluation results can depend on the verification dataset being used.

### 3. Generate Scorecard and Visualize
After making all selections, the user clicks the **"Generate Scorecard"** button.

The platform processes the selected options and displays the evaluation results immediately.

#### Primary Visualization
A line graph is generated showing:

- the chosen metric on the **y-axis**
- the forecast lead time on the **x-axis**

Each selected model is represented by a separate line, allowing direct visual comparison of performance over time.

#### Summary Table
Below the graph, the platform displays a table with precise metric values for each selected model at key forecast lead times, for example:

- Day 3
- Day 5
- Day 7
- Day 10

This helps users move from visual interpretation to exact numerical comparison.

### 4. Guided Analysis and Interpretation
To support learning, the platform displays dynamic guiding questions based on the user’s current selections.

Examples include:

- **Which model maintains the highest ACC for Z500 beyond Day 7?**
- **If you switch to Total Precipitation and CRPS, does the top-performing model change?**
- **Why might CRPS rank models differently than RMSE for precipitation?**
- **At what lead time do AI models generally become more skillful than IFS-HRES?**

These questions encourage users to think critically about the meaning of the results and the implications of different evaluation choices.

### 5. Explore and Repeat
The user is encouraged to repeat the process by changing:

- the metric
- the variable
- the region
- the selected models
- the ground-truth dataset

This repeated exploration helps build intuition about model evaluation and forecast verification.

---

## Key Concepts Covered

This demo introduces several important ideas in forecast evaluation:

### RMSE
A deterministic metric that measures the average size of forecast errors.  
It is easy to interpret, but large errors strongly affect the score.

### ACC
A pattern-based metric that measures how well the forecast captures the large-scale spatial pattern compared to observations.  
It is especially useful for variables like geopotential height.

### CRPS
A probabilistic metric that evaluates the full predictive distribution rather than only a single deterministic value.  
It is particularly helpful when working with uncertain or highly variable fields like precipitation.

### SEEPS
An event-based metric that is often used for precipitation verification and focuses on categorical forecast quality.

### Ground Truth Selection
The demo also shows that the choice of reference dataset matters.  
For example, precipitation verification may look different depending on whether the forecast is compared against **ERA5**, **IMERG**, or **CHIRPS**.

---

## What I Did

For **Demo 3**, I worked on documenting and presenting an **interactive forecast verification workflow** delivered through the **Genevieve website** in collaboration with **Rhiza**.

This demo was designed to move beyond model execution and focus on **evaluation and comparison**. The platform allows users to benchmark forecasting models in a hands-on way by selecting metrics, variables, models, regions, and ground-truth datasets, then generating plots and scorecards for interpretation.

The main contribution of this demo is creating a learning experience where participants do not just see results, but also understand **how those results are measured and interpreted**.

---

## Technologies and Tools

- Google Colab
- Interactive online platform / website
- Genevieve website
- Forecast verification metrics
- Comparative visualizations and scorecards

---

## Why This Demo Is Important

This demo is important because evaluating a forecast model is just as critical as running it.

A model may perform well under one metric but not another. It may also perform differently depending on the variable, lead time, or ground-truth dataset. By giving users the ability to interact with these dimensions directly, the demo helps build a deeper understanding of **forecast skill**, **model benchmarking**, and **scientific interpretation**.

---

## Conclusion

**Demo 3** is an interactive model evaluation and comparison tool that teaches users how weather forecasting systems are benchmarked using professional verification metrics.

By combining explanation, visualization, and guided analysis, the demo helps participants understand not only **which model performs better**, but also **why performance depends on the evaluation method and data choices**.

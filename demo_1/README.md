# Demo 1 - Environment Checker

## Overview
Demo 1 is a Python application built with Streamlit that launches a local interactive web page to help users verify that their environment is correctly configured for the course.

The app acts as a guided checklist and helps users confirm that the required tools, libraries, and system settings are working properly.

## What this demo does
The demo guides the user through the following checks:

1. **Launch the Environment Checker**  
   The user runs a Python script from the terminal, which opens a local Streamlit web page.

2. **System Health Check**  
   The app checks system information such as:
   - Operating System
   - Available RAM
   - GPU presence

3. **WSL and GPU Driver Verification**  
   The user opens a WSL terminal and runs:

   ```bash
   nvidia-smi

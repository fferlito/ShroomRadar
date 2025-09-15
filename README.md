# 🍄 ShroomRadar: A Machine Learning Framework for Predicting Mushroom Occurrences

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Abstract

ShroomRadar is a machine learning project dedicated to predicting the probability of mushroom species presence based on a variety of environmental factors. The core of this project is a `GradientBoostingClassifier` model trained on publicly available geospatial and climate data. The system is designed to generate daily probability maps for user-defined areas of interest, providing a valuable tool for ecological monitoring and mycology enthusiasts. The repository includes the complete data science workflow—from data acquisition and preprocessing to model training—as well as a containerized production pipeline for automated daily inference.

## Table of Contents

- [Project Overview](#project-overview)
  - [Data Sources](#data-sources)
  - [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
  - [1. Data Preparation and Model Training](#1-data-preparation-and-model-training)
  - [2. Inference Pipeline](#2-inference-pipeline)
- [Outputs](#outputs)

## Project Overview

This project provides an end-to-end framework for predicting mushroom occurrences. It integrates multiple data sources to capture the complex interplay of environmental conditions that influence fungal growth.

### Data Sources

The model is trained on a combination of biotic, climatic, topographic, and soil-related data:

-   **Mushroom Occurrences**: Positive samples are sourced from iNaturalist (`fetch_mush_data.ipynb`). Negative samples are generated randomly across the study area (`make_negative.ipynb`).
-   **Climate Data**: Historical and near-real-time weather data (e.g., temperature, precipitation) are fetched from the Open-Meteo API (`get_climate_data.ipynb`).
-   **Topography**: Elevation and aspect data are derived from NASA's Shuttle Radar Topography Mission (SRTM) Version 3 product (`append_elevation.ipynb`).
-   **Land Cover**: Land use and land cover information is sourced from the Corine Land Cover dataset (`LC.ipynb`).
-   **Soil Composition**: Soil properties are obtained from SoilGrids (`soilgrid.ipynb`).

### Methodology

The workflow is divided into two main phases: model training and production inference.

1.  **Data Preprocessing**: All data sources are cleaned, aligned, and integrated into a unified dataset. This involves linking climate and environmental data to each mushroom occurrence point (or negative sample point).
2.  **Model Training**: A `GradientBoostingClassifier` is trained on the preprocessed dataset to learn the relationship between the environmental features and mushroom presence (`train_model.ipynb`).
3.  **Inference**: A containerized pipeline automates the daily generation of prediction maps. It prepares an inference grid, fetches the latest climate data, runs the model to predict presence probabilities, and uploads the final map to Ellipsis Drive.

## Repository Structure

The repository is organized as follows:

```
ShroomRadar/
│
├── .env                  # Environment variables
├── data/                 # Raw and processed data files
├── notebooks/            # Jupyter notebooks for data processing and modeling
├── docker/               # Dockerized production pipeline
│   ├── main.py           # Main script for orchestrating the pipeline
│   └── ...
├── assets/               # Images and static assets for documentation
└── README.md             # Project documentation
```

## Installation

To set up the project locally, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/ShroomRadar.git
cd ShroomRadar
pip install -r requirements.txt
```

For the production pipeline, Docker and Docker Compose are required. Please refer to the documentation in the `docker/` directory for detailed setup instructions.

## Usage Workflow

The project is structured around two key workflows: training a new model and running the inference pipeline.

### 1. Data Preparation and Model Training

The Jupyter notebooks in the `notebooks/` directory guide you through the process of preparing data and training the model. The primary steps are:

-   **Data Acquisition**:
    -   `fetch_mush_data.ipynb`: Download mushroom species data from iNaturalist.
    -   `make_negative.ipynb`: Generate negative samples for training.
-   **Feature Engineering**:
    -   `get_climate_data.ipynb`: Download historical climate data for each sample.
    -   `append_elevation.ipynb`: Append elevation and aspect data.
    -   `LC.ipynb`: Append Corine Land Cover data.
    -   `soilgrid.ipynb`: Append SoilGrids data.
-   **Model Training**:
    -   `train_model.ipynb`: Train the `GradientBoostingClassifier` on the final dataset.

### 2. Inference Pipeline

The inference process generates daily mushroom presence predictions for a specified area of interest.

#### Preparing the Grid for Inference

Before running the main pipeline, a gridded GeoJSON of the area of interest must be prepared with static environmental data.

1.  **Generate Grid** 🌐: Use `grid.ipynb` to create a gridded GeoJSON file from an input geometry.

| Input (`spain-provinces.geojson`) | Output (`spain_grid_3km.geojson`) |
| :-------------------------------- | :-------------------------------- |
| <img src="assets/map.png" alt="drawing" width="300"/> | <img src="assets/grid.png" alt="drawing" width="300"/> |

2.  **Append Static Data**: Enrich the grid with static environmental data using the same notebooks as in the training workflow (`append_elevation.ipynb`, `LC.ipynb`, `soilgrid.ipynb`).

#### Production Inference

The production pipeline is containerized using Docker and orchestrated by `docker/main.py`. It takes the pre-prepared gridded GeoJSON and performs the following steps automatically:

1.  **Append Near-Real-Time Climate Data**: Fetches and appends the latest 14-day climate data from the Open-Meteo API.
2.  **Generate Predictions**: Loads the trained model and computes the presence probability for each grid cell.
3.  **Filter and Post-process**: Filters the predictions to retain locations with a probability score above a defined threshold.
4.  **Upload Results**: Uploads the final GeoJSON prediction map to a designated service, such as Ellipsis Drive.

## Outputs

The primary output of this project is a GeoJSON file representing the daily probability of mushroom presence for a given area. Each feature in the GeoJSON corresponds to a grid cell and includes a `prediction_score` attribute.

# 🍄 ShroomRadar 🌎


## Dataset

### Mushroom data 🍄

- *Mushrooms location and timestamps*: use the `fetch_mush_data.ipynb` notebook to download a csv with the location and timestamp of a mushroom species, using the iNaturalist api. 
- *Negative samples and timestamps*: use the `make_negative.ipynb` notebook to generate a csv with random locations and times, to be used as negative classes for the training


### Climate data 🌧️

- `get_climate_data.ipynb` contains the code necessary to download the climate data relative to the mushroom data obtained in the previous section
- `append_climate_data.ipynb`: code to append the climate data to the geometry geojson

### Environmental data ⛰️

- *Elevation data (NASA SRTM V3 product)* : `append_elevation.ipynb`
- *Corine map*: `LC.ipynb` 

## Training code

- *GradientBoostingClassifier*: code to train the model is in `train_model.ipynb`


## Inference

In order to get new predictions out of the model, we need to fetch the climate and environment data for the region of interest. The repository contains code to generate a heatmap that shows the probability distribution for a specific day, using as input a gridded geometry.

### Gridded map 🌐

- `grid.ipynb` contains the code to generate a gridded geojson. 

Input geometries (spain-provinces.geojson)   |  Output (spain_grid_3km.geojson)
:-------------------------:|:-------------------------:
<img src="assets/map.png" alt="drawing" width="300"/>  |  <img src="assets/grid.png" alt="drawing" width="300"/>


### Attach the environmental data

### Attach the climate data

### Inference


# 🍄🌎 ShroomRadar


## Dataset

### Mushroom data

- *Mushrooms location and timestamps*: use the `fetch_mush_data.ipynb` notebook to download a csv with the location and timestamp of a mushroom species, using the iNaturalist api. 
- *Negative samples and timestamps*: use the `make_negative.ipynb` notebook to generate a csv with random locations and times, to be used as negative classes for the training

### Environmental data

- *Elevation data (NASA SRTM V3 product)* : `append_elevation.ipynb`


### Training code

- *GradientBoostingClassifier*: code to train the model is in `train_model.ipynb`
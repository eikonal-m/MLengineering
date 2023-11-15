# MLengineering: Refresher and udpates

## Defining the model factory

 - training system: automate the training of models 
 - model store: persist successfully trained models
 - drift detector: detect changes in model performance

these, combined with the deployed "prediction system" create the model factory

### Training design options

**1/ Train Run**
perform training and prediction in the same process

**2/ Train Persist**
training runs in batch, with prediction reading the trained model from a model store



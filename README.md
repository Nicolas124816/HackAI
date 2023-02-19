# HackAI

This project takes data one wine, uses existing data on quality of the wine, and builds a model to take in new data on wine and predict the quality from it.

TAKING IN DATA
This project imports all of the wine data from Keggle

CLEANING DATA
All tuples of wine data are removed if they contain a figure that is an outlier in the set.

TRAINING
The model is trained with the data with two different methods.
LukePredict uses a tensor model with two nodes to model the data.
SwuPredict uses SKLearn to generate a model for new data.

TESTING
With the two possible generated models, data can more accuraly predict unknown wine qualities.
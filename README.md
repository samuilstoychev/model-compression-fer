# L44-Mini-Project

The project repository has the following structures: 
* `ckplus_labels.csv` includes the manual annotations for the CK+ dataset. 
* `fer_model.py` defines the baseline model architecture. 
* `data.py` and `evaluation.py` define auxiliary functions for loading the CK+ dataset and evaluating models. 
* `/logs` includes all logs produced during evaluation. You can load them using Python's `pickle` library.
* `/weights` includes the weights generated during training the baseline or fine-tuning pruning or weight clustering. You can load them using Keras' `load_weights` function. 

Also, the following notebooks have been created for running the experiments: 
* `ModelTraining.ipynb` trains the baseline model and stores its weights. 
* `Quantisation.ipynb`, `Pruning.ipynb` and `WeightClustering.ipynb` include the implementations of the three compression strategies for the project. 
* `ResultsAnalysis.ipynb` generates tables and visualisations of the results based on the logs generated from the previous three notebooks. Figures are stored in the `/figures` folder. 

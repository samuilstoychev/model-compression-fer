# The Effect of Model Compression on Fairness in Facial Expression Recognition

This repository includes the code for the experiments presented in *"The Effect of Model Compression on Fairness in Facial Expression Recognition"* (https://arxiv.org/abs/2201.01709) - to be presented at [AMAR 2022](https://cse.usf.edu/~tjneal/AMAR2022/index.html). 

## Contents 

The project repository has the following structure: 
* `ckplus_labels.csv` includes the manual annotations for the CK+ dataset. 
* `fer_model.py` defines the baseline model architecture. 
* `data.py` and `evaluation.py` define auxiliary functions for loading the CK+ dataset and evaluating models. 
* `/logs` includes all logs produced during evaluation. You can load them using Python's `pickle` library.
* `/weights` includes the weights generated during training the baseline or fine-tuning pruning or weight clustering. You can load them using Keras' `load_weights` function. 

Also, the following notebooks have been created for running the experiments: 
* `ModelTraining.ipynb` trains the baseline model and stores its weights. 
* `Quantisation.ipynb`, `Pruning.ipynb` and `WeightClustering.ipynb` include the implementations of the three compression strategies for the project. 
* `ResultsAnalysis.ipynb` generates tables and visualisations of the results based on the logs generated from the previous three notebooks. Figures are stored in the `/figures` folder. 

Notebooks appended with `-RAFDB` include the experiments conducted on the RAF-DB dataset (identical to the ones originally conducted on CK+). 

## Citation
Please cite our paper in your publications if this code helps your research.
```
@article{stoychev2022effect,
  title={The effect of model compression on fairness in facial expression recognition},
  author={Stoychev, Samuil and Gunes, Hatice},
  journal={arXiv preprint arXiv:2201.01709},
  year={2022}
}
```

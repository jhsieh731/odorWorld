# Odor World: Odor Encoding and Demixing

This project uses machine learning to explore the decodability of odor mixtures.

## Repository Structure

``` {}
|_ figures/
|_ uniform/
   |_ fixed_uniform_generator.ipynb
   |_ model.ipynb
   |_ dataset.py
   |_ binary_solver.py
|_ gaussian/
   |_ gaussian_data_generator.ipynb
   |_ model.ipynb
   |_ dataset.py
|_ pipeline/
   |_ checkpoints/
   |_ data/
   |_ epoch_stats/
   |_ final_models/
   |_ indices/
   |_ intensity_generator.ipynb
   |_ model.ipynb
   |_ plots.ipynb
   |_ dataset.py
   |_ rsample_dataset.py
   |_ run.py
   |_ run.sh
|_ scraper.ipynb
|_ data.csv
|_ data_chemicals.csv
|_ data_binary.csv
|_ data_final.csv
|_ molecules.csv
|_ binary_opens.pkl
```

The project structure is split into three main directories: `uniform`, `gaussian`, and `pipeline`. The `figures` directory` is a catch-all for images generated during analysis, and is not complete (i.e. please ignore).

Each of the other three directories are analagous in structure, with the `pipeline` designed for continued future use; the `uniform` and `gaussian` directories are early iterations. These each have a `%data_generator.ipynb` notebook as well as a `model.ipynb` notebook. The data generator is used to create the `.csv` and `.pkl` files used in training, while the model is used to train the neural network models. Additionally, `%dataset.py` defines the custom dataset for PyTorch training.

## Data Files

These are in the main directory. Most data is read from `binary_opens.pkl`, and `molecules.csv` contains additional information that may be used in future steps. Most parts of the scraper are in `scraper.ipynb`, though the cells may not be fully representative of the existing files.

## Pipeline

**Labeling.** Example: `I_500_M3` is a _noisy fixed intensity_ encoding with _500_ possible odor entity labels and with a _mixture of size 3_.

- Encodings:
  - `U`: noisy uniform
  - `I`: noisy fixed intensity
  - `N`: multivariate normal/Gaussian
- `Mx`: Mixtures of size $M=x$

### Subdirectories and files

- `checkpoints/` stores the model at `cpx` where $x$ denotes the epoch number. Each checkpoint can be unpacked into: epoch number, model states, optimizer states, loss, and accuracy.
- `data/` stores the pickled data files for each encoding. The number denotes the number of possible entity labels.
- `epoch_stats/` stores the `.csv` files with the loss and accuracies for each encoding/mixture so that we can plot training history.
- `final_models/` stores the converged models for each encoding/mixture. The name of each model also stores the total number of entity labels.
- `indices/` stores the indices to the data files to allow for fast lookup through the custom dataset/dataloader.
- `intensity_generator.ipynb` creates the pickled data file for the noisy fixed intensity encoding.
- `dataset.py` is a custom dataset that loads all data for a given dataset.
- `rsample_dataset.py` is a custom dataset that loads random samples from a given dataset, and allows training for extremely large datasets.
- `model.ipynb` is a notebook used to train models.
- `plots.ipynb` is a notebook used to conduct analyses on models and data.
- `run.py` is the script used to train models on the cluster.
- `run.sh` is the bash file run on the cluster.

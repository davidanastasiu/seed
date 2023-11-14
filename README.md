## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.8.3. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create --name seed python=3.8.3
conda activate seed
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch
python -m pip install -r requirements.txt
```

## Files organization

Download the datasets from [here](https://clp.engr.scu.edu/static/datasets/seed_datasets.zip) and upzip the files in the `data_provider` directory.
You should now have a `data_provider/datasets` directory containing 4 stream sensor (file names end with _S_fixed.csv) and 4 rain sensor (file names end with _R_fixed.csv) datasets.

Use parameter '--val_size' to set the number of randomly sampled validation points which will be used in the training process. 

Use parameter '--train_volume' to set the number of randomly sampled validation points which will be used in the training process. 

Use parameter '--sub_mean_threshold' to set the threshold which will be used in the sample-wise oversampling process. 

Use parameter '--times' to set the step size which will be used in the sample-wise oversampling process. 

Refer to the annotations in 'options.py' for other parameters setting.


## Training mode

run 'main.ipynb' with model.train()

## Inference mode

The inference has already been included in 'main.ipynb'. You can also run 'main.ipynb' without model.train() for inferencing only.

## Future improvement

As can be seen, the code has not been cleaned up for "production" due to lack of time. It includes some parameters that were used in other projects. This will be improved when time allows.


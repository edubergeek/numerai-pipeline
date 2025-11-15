# numerai-pipeline
Project: Numerai Pipeline
Author:  Curt Dodds
Date:    2024-09-24
Github:  https://github.com/edubergeek/numerai-pipeline/

## Getting Started
Linux bash scripts are provided to run. 
On Windows it is recommended to install Docker for Desktop and git bash.
This provides an equivalent runtime command line environment (CLI).
Fear not. You can easily launch a Jupyter notebook from the CLI.

### Download
Run the prepare.sh script to download (using Python) the data files from the Numerai website.

### ETL
Once the data is on your local system you can prepare it for training your models.

The next step in the prepare.sh script is to run the Python program "etl.py".
This program transforms the Numerai parquet files into TensorFlow shard files.
These shard files can be efficiently loaded into one or more GPUs by Keras and TensorFlow.
Options are provided to split the data into train/validation/test datasets.
You can also set the number of training examples to store in each shard file.
Finally you can choose to preserve era information in each shard file name or not.
This can be useful for dynamically training on certains eras (e.g. even or odd or every 4th era).
There is an experimental option to balance examples by using extra eras rows to augment the underrepresented 0/5 and 1/4 examples.
Use this at your own risk!

### Generate a model
Run the Python program 'naiMLP.py' to generate a Keras neural network having no weights.
This creates a directory in a directory having the same name as the model, e.g. 'mlp'.
You can create your own "model zoo" and see which ones perform the best in the tournament.
Once you have a model you can train it.

### Train the model(s)
Use the bash script 'train.sh' to train one or more models.
Edit the embedded configuration parameters for your model then run the script to begin training.
If you run in Jupyter be sure to set the IS_JUPYTER flag in predict.py to True.
If running from the command line the IS_JUPYTER flag needs to be False.
Because training can run a long time I usually run from the command line but Jupyter is useful in dev mode.
By using X11 forwarding with ssh (Putty and XMing on Windows) I get a nice training plot. 
I added Tensorboard support so that's a useful option if not useing X11 forwarding.


The default behavior is to only save weights when the validation loss improves.
The script "bestepoch.sh" can be run after training is complete.
It will select the training epoch that had the lowest validation loss.
Obviously this is just the highest numbered epoch so it's a convenience script.

## Usage
-    bash prepare.sh - download training data and train models
-    bash predict.sh - download live data, predict and submit predictions

## Release Notes
-    1.0 - added support for ATLAS data release
-    1.3 - added support for autoencoding classifier and force download

## Todo
There are endless improvements to be made to any software project.
Here are some I have in mind:
-  Train decision tree models (XGBoost, LGBM) from TFRecord files
-  GPU accelerated decision tree models (maybe using NVIDIA RAPIDS)


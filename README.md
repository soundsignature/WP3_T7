# WP3_T7
Design and implementation of a pipeline that:
1. **Process**, given user specifications, the data generated by the WP2_T3 pipeline to **create the clips to train the AI models**.
2. **Implements** the three selected **AI models** (Vggish, PaSST, EffAT) and well as an **standardized way** of **training, testing and performing inference** with them.

The recommended Python version is **3.9.13**

The project uses [Poetry](https://python-poetry.org/) for dependency management and packaging. Before running any code, ensure Poetry is installed and configured properly:

- **Windows users**: Use the command `poetry install --with intel`
- **Linux users**: Use the command `poetry install --without intel`

The recommended Poetry version is **1.8.3**

## Repository Structure

The repository is organized as follows:

1. **scripts/**: Contains scripts for training, testing, and inference.
   - **`train.py`**: Script to train the AI models.
   - **`test.py`**: Script to test the trained models.
   - **`inference.py`**: Script to perform inference using the trained models.

2. **src/**:
   - **config/**: Contains all the `.yaml` files that define modifiable parameters for training and testing each AI model.
   - **models/**: Houses implementations and architectures of the three models:
     - **`effat_model.py`**: Implementation of the EfficientAT model for training, testing, and inference ([EfficientAT repo](https://github.com/fschmid56/EfficientAT)).
     - **`passt_model.py`**: Implementation of the PaSST model for training, testing, and inference ([PaSST repo](https://github.com/kkoutini/PaSST)).
     - **`vggish_model.py`**: Implementation of the Vggish model for training, testing, and inference ([Vggish repo](https://github.com/tensorflow/models/blob/master/research/audioset/vggish)).
   - **pipeline/**:
     - **`dataset.py`**: Contains the `EcossDataset` class, responsible for modeling and processing datasets.
     - **`utils.py`**: Contains utility functions and helper classes used within the `EcossDataset` class and other modules related to the AI models.

3. **`poetry.lock`**: A file generated by Poetry, locking in the exact versions of all dependencies.

4. **`pyproject.toml`**: Defines the project's metadata, including required packages, their versions, and the compatible Python versions.

## Configure enviroment variables

To set up the necessary variables to run the code, create a file named **`.env`** with the following content:
```
# Common Parameters
EXP_NAME = # desired name of the folder to store the results

# Dataset Management parameters
NEW_ONTOLOGY = 'Ship,Biological,...' # Top level to group labels
UNWANTED_LABELS = 'Tursiops,SpermWhale,...' # Labels to drop
TEST_SIZE = # Size of the test set in decimal format (e.g. 0.3)

# Train parameters
ANNOTATIONS_PATHS= 'path/to/data1,path/to/data1,...'
YAML_PATH = path/to/yaml/config/file
MODEL_TYPE = # effat passt or vggish

# Inference / Test parameters
INFERENCE_DATA_PATH = path/to/file/to/predict
PATH_MODEL_TEST = path/to/trained/model/weights

# Check labels
DATASET_PATH_CHECK = path/to/dataset/to/analyze
YAML_LABELS_CHECK = path/to/specific/yaml
STORE_PATH_CHECK = path/where/the/labels/are/stored

```
--------------------------------------------------------------



## POETRY FIRST TIME:
1. Install poetry on your base python virtual environment with "pip install poetry".
2. Write "poetry init" in order to create the project.
3. In the pyproject.toml file change the name parameter in [tool.poetry] for the name "pipeline".
4. Write "poetry config virtualenvs.in-project true". This will tell poetry to locate the .venv file that will be created in the root of the project.
5. Write "poetry install". This will create the .venv file and install all the packages and dependecies needed for your project.
6. In case you want to add more, simply write "poetry add package-name".

## POETRY AFTER FIRST CONFIGURATION:
1. Install poetry on your base python virtual environment with "pip install poetry".
2. Write "poetry config virtualenvs.in-project true". This will tell poetry to locate the .venv file that will be created in the root of the project.
3. Write "poetry install". This will create the .venv file and install all the packages and dependecies needed for your project.
4. In case you want to add dependecies and package, write "poetry add package-name".

## HOW TO RUN YOUR CODE WITH YOUR POETRY VENV?
Two ways:
1. Write "poetry shell". This will activate the venv and you will be able to navigate to your script and do "python X.py". In order to exit this shell write "exit".
2. Activate the venv manually.


## VGGISH POETRY
If you use linux, then you dont need the package tensorflow-intel because it will raise an error. Install everything like this: "poetry install --without intel".
If you use windows, then you will need the package tensorflow-intel because otherwise you will experience errors. Install everything like this: "poetry install --with intel".

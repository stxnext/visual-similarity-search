# Visual Similarity Search - Category-based Image Comparison

Visual Similarity Search Engine demo app - built with the use of PyTorch Metric Learning and Qdrant vector database.
Similarity search engine is used for comparing uploaded images with content of selected categories. 
There are two modules created within the engine:
1. Interactive Application - used for finding the closest match of uploaded or selected image within a given data category.
2. Model Training/Deployment Module - used when a new data category is added to the application.

**Demo: [Visual Similarity Search App](https://visual-search.stxnext.pl/)**

Proudly developed by [STX Next Machine Learning Team](https://www.stxnext.com/services/machine-learning/?utm_source=github&utm_medium=mlde&utm_campaign=visearch-demo)

## Table of Contents
  - [Installation](#installation)
    - [Installation - Local - Manual](#installation-local-manual)
    - [Installation - Local - Docker](#installation-local-docker)
    - [Installation - Cloud - Docker](#installation-cloud-docker)
    - [Installation - Accessing MinIO](#installation-accessing-minio)
    - [Installation - Docker Compose Structure](#installation-docker-compose-structure)
  - [Datasets](#datasets)
    - [Current Datasets](#datasets-current)
    - [Queued Datasets](#datasets-queued)
  - [Application](#application)
    - [Add or Update Data](#add-or-update-data)
    - [Model Training Module](#model-training-module)
    - [Training Results](#training-results)
    - [Qdrant Database Update](#qdrant-database-update)
  - [Using Jupyter Notebooks](#using-jupyter-notebooks)
  - [Installation Dependencies and Other Issues](#installation-dependencies-and-other-issues)
  - [Authors](#Authors)
  - [Licences](#licences)


## Installation

Both modules mentioned in the introduction use libraries specified in the ***poetry.lock*** file which are resolved
based on the contents of ***pyproject.toml*** file.

Installation and functioning of the modules depends on the `data` folder and two environment files - first for Docker-Compose build, 
and second for working of Python app.

Environment variables file for Docker-Compose is ***.env***. It contains a selection of variables:
  * `QDRANT_PORT` - port for Qdrant service,
  * `INTERACTIVE_PORT` - port for Streamlit service,
  * `PYTHON_VERSION` - used Python version,
  * `QDRANT_VERSION` - version of Qdrant's docker image,
  * `INTERACTIVE_APP_NAME` - name of docker image's working directory,
  * `QDRANT_VOLUME_DIR` - Qdrant container's volume directory for Qdrant's storage,
  * `MODEL_VOLUME_DIR` - interactive container's volume directory for local pull of models from cloud storage,

Environment variables file for Python processing is ***.env-local*** or ***.env-cloud***. It contains a selection of variables:
  * `QDRANT_HOST` - host for Qdrant service,
  * `MINIO_HOST` - host for MinIO S3 cloud storage,
  * `MINIO_ACCESS_KEY` - access key for MinIO S3 cloud storage,
  * `MINIO_SECRET_KEY` - secret key for MinIO S3 cloud storage,
  * `MINIO_BUCKET_NAME` - default bucket name in MinIO S3 cloud storage,
  * `MINIO_MAIN_PATH` - MinIO object path to directory containing `data` folder,
  * `TYPE` - environment type (options for cloud: PROD, TEST, DEV | options for local: LOCAL).

Apart from environmental variables, application uses contents of the dedicated `data` folder structure (available on the same level as ***.env*** file:
```angular2html
api
common
data
├── metric_datasets
│   ├── dogs
│   ├── shoes
│   ├── celebrities
│   ├── logos
├── models
│   ├── dogs
│   └── shoes
│   └── celebrities
│   └── logos
└── qdrant_storage
    ├── aliases
    ├── collections
    │   ├── dogs
    │   └── shoes
    │   └── celebrities
    │   └── logos
    └── collections_meta_wal
interactive
metrics
notebooks
scripts

```
The structure of the `data` folder is split as follows:
  * `metric_datasets` - split into folders corresponding with data categories, each containing raw pictures that were used for model training and are being pulled as a result of visual search.
  * `models` - split into folders corresponding with data categories, each containing pretrained deep learning models,
  * `qdrant_storage` - storage for vector search engine (Qdrant), each data category has its own collection.

### Local - Manual

Installation using the terminal window:
* Install ***git***, ***docker*** packages.
* `cd` to your target directory.
* Clone [repository](https://github.com/stxnext/visual-similarity-search) (preferably use SSH cloning).
* Download `data.zip` and some data files using following links.
  * [data.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/data.zip) - template for directory tree with initial Qdrant structure.
  * [celebrities.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/celebrities.zip) - metadata, models and image repository.
  * [dogs.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/dogs.zip) - metadata, models and image repository.
  * [logos.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/logos.zip) - metadata, models and image repository.
  * [shoes.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/shoes.zip) - metadata, models and image repository.
  * [waste.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/waste.zip) - metadata, models and image repository.
* Unpack selected datasets to the cloned repository so that the folder structure from previous section is retained.
* In the `metrics/consts.py` in the definition of `MetricCollections` class comment dataset names that were not added:
```python
class MetricCollections(Enum):
    """
    Enum of available collections and pretrained models for similarity.
    """

    DOGS = "dogs"
    SHOES = "shoes"
    CELEBRITIES = "celebrities"
    LOGOS = "logos"
    WASTE = "waste"
```
* Install Python version 3.10 and ***pip***, ***pipenv*** libraries.
```
sudo apt-get install python3.10 
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
python3.10 -m pip install pipenv  
```
* Set up local environment and run shell:
```
python3.10 -m pipenv --python 3.10
python3.10 -m pipenv shell
```
* Within the shell install [poetry](https://pypi.org/project/poetry/) and dependencies:
```
pip install poetry --no-cache 
poetry update 
```
* Run docker and set up Qdrant database docker image
```
docker run -p 6333:6333 \                               
    -v $(pwd)/data/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:v0.10.3

```
* Create ***.env-local-no-docker*** file by copying and renaming the ***.env-local*** file.
* Fill parameters of ***.env-local-no-docker*** file with specific values:
  * `QDRANT_HOST=localhost`,
* Load environmental variables.
```
export PYTHONPATH="${PYTHONPATH}:/"
export $(grep -v '^#' .env | xargs)
export $(grep -v '^#' .env-local-no-docker | xargs)
```
* Run Streamlit app
```
# If Poetry env set as default Python env.
streamlit run interactive/search_app.py --server.port=$INTERACTIVE_PORT --server.address=0.0.0.0

# Otherwise.
poetry run python -m streamlit run interactive/search_app.py --server.port=$INTERACTIVE_PORT --server.address=0.0.0.0
```
* Access the visual similarity search engine under URL: [localhost](http://0.0.0.0:8080).

### Local - Docker

Installation using the terminal window:
* Install ***git***, ***docker***, ***docker-compose*** and ***make*** packages.
* `cd` to your target directory.
* Clone [repository](https://github.com/stxnext/visual-similarity-search) (preferably use SSH cloning).
* Download `data.zip` and some data files using following links.
  * [data.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/data.zip) - template for directory tree with initial Qdrant structure.
  * [celebrities.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/celebrities.zip) - metadata, models and image repository.
  * [dogs.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/dogs.zip) - metadata, models and image repository.
  * [logos.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/logos.zip) - metadata, models and image repository.
  * [shoes.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/shoes.zip) - metadata, models and image repository.
  * [waste.zip](https://storage.googleapis.com/stx-ml-public/Visual-Similarity-Search/Data/waste.zip) - metadata, models and image repository.
* Unpack selected datasets to the cloned repository so that the folder structure from previous section is retained. 
* In the `metrics/consts.py` in the definition of `MetricCollections` class comment dataset names that were not added:
```python
class MetricCollections(Enum):
    """
    Enum of available collections and pretrained models for similarity.
    """

    DOGS = "dogs"
    SHOES = "shoes"
    CELEBRITIES = "celebrities"
    LOGOS = "logos"
    WASTE = "waste"
```
* To set up a dockerized application, execute one of the options below in the terminal window.
``` 
# Use Makefile:
make run-local-build 

# Optional:
make run-local-build-qdrant-restart
make run-local-build-interactive-restart
```
* Access the visual similarity search engine under URL: [localhost](http://0.0.0.0:8080).

### Cloud - Docker

Installation using the terminal window:
* Install ***git***, ***docker***, ***docker-compose*** and ***make*** packages.
* `cd` to your target directory.
* Clone [repository](https://github.com/stxnext/visual-similarity-search) (preferably use SSH cloning).
* Create ***.env-cloud*** file by copying and renaming the ***.env-local*** file.
* Fill parameters of ***.env-cloud*** file with specific values:
  * `QDRANT_HOST=qdrant-cloud`,
  * `MINIO_HOST`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME` with MinIO-specific data,
  * `MINIO_MAIN_PATH` with path to directory containing `data` folder on MinIO's `MINIO_BUCKET_NAME`,
  * `TYPE=DEV` is preferred over TEST and PROD (option LOCAL does not work with cloud).
* To install new environment execute one of the options below.
``` 
# Use Makefile - run one at the time:
make run-cloud-build

# Verify if run-cloud-build ended using logs in interactive-cloud container. Then, run the following two.
make run-cloud-build-qdrant-restart
make run-cloud-build-interactive-restart
```
* Access the visual similarity search engine under URL: [localhost](http://0.0.0.0:8080).

### Accessing MinIO

Current implementation allows you to access category-related datasets from the level of MinIO cloud storage. 
All communication between a storage and an application/Docker is performed via the MinIO Python client.
For secret and access keys contact the MinIO service's administrator or create a 
[service account](https://min.io/docs/minio/linux/administration/identity-access-management/policy-based-access-control.html)
for your bucket. This may be performed from the level of MinIO Console.

A need for building other connectors may arise - for now only manual fix could be applied: 
* Replace the client's definition and adjust functions for getting/listing objects.

### Docker Compose Structure

There are two compose files, each responsible for setting up a different way of data provisioning to the final
application:
* `docker-compose-local.yaml` - After the `data` folder is manually pulled by the user, compose file creates two services: `qdrant-local` and `interactive-local`, which share appropriate parts of the `data` folder as their respective volumes.
* `docker-compose-cloud.yaml` - The `data` folder is available on the MinIO cloud storage with access via the Python client. Only Qdrant-related and Model-related data is pulled locally for the services to run properly. Compose file creates two services: `qdrant-cloud` and `interactive-cloud` which share `model_volume` and `qdrant_volume` volumes.

Those files share Qdrant and Python versions, `.env` file inputs, `Dockerfile-interactive` file and `docker-entrypoint-interactive.sh` script.


## Datasets

Both Model Training and Application modules use the same scope of datasets. 
Each category corresponds with a single dataset.
Models are trained separately for each category. 
Application returns search results from the scope of images available only within the selected data category.
All datasets listed below are the property of their respective owners and are used .

### Current Datasets 

* List of datasets with trained models that are available in the Visual Similrity Search application:
  * [Shoes dataset](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k)
    * A large shoe dataset consisting of 50,025 catalog images collected from Zappos.com. 
    * The images are divided into 4 major categories — shoes, sandals, slippers, and boots — followed by functional types and individual brands. 
    * The shoes are centered on a white background and pictured in the same orientation for convenient analysis.
  * [Dogs dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
    * The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. 
    * This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization.
  * [Celebrities dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
    * A large dataset containing images of list of the most popular 100,000 actors as listed on the IMDb website (in 2015) together with information about their profiles date of birth, name, gender.
    * Since multiple people were present in original pictures, many of the cropped images have wrong labels. This issue was mostly resolved by selecting images of size +30kB.
    * Only pictures available in RGB mode were selected.
  * [Logos dataset](https://www.kaggle.com/datasets/lyly99/logodet3k)
    * The largest logo detection dataset with full annotation, which has 3,000 logo categories, about 200,000 manually annotated logo objects and 158,652 images.
  * [Waste dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
    * A large household waste dataset with 15,150 images from 12 different classes of household garbage; paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, and trash.

### Queued Datasets

* List of datasets that are queued for implementation:
  * [Fashion dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
    * Thr growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon. In addition to professionally shot high resolution product images, we also have multiple label attributes describing the product which was manually entered while cataloging. 


## Application

The public version of the Cloud-based application is available here: [Visual Similarity Search App](https://visual-search.stxnext.pl/).
Frontend is written in [Streamlit](https://streamlit.io/) and uses dedicated assets, local/cloud image storage, 
pre-built models and Qdrant embeddings. The main page is split into 4 main sections:
* `Input Options` - Initial category selection via buttons and option for resetting all inputs on the page.
* `Business Usage` - Dataset description and potential use cases.
* `Image Provisioning Options` - Chooses a way of selecting an input image.
* `Input Image` - Shows a selected image.
* `Search Options` - Allows a selection of a similarity benchmark and a number of shown images. After the search, you can reset the result with a dedicated button. Images that are the most similar to the input image appear in this section.
* `Credits` - General information about repository.

A given section is visible only when all inputs in previous sections were filled.


## Add or Update Data

A new dataset can be added to the existing list of options by:
* Preprocessing the new/updated dataset and adding it to the `data` folder.
* Training embedding and trunk models.
* Uploading training results to the [Tensorboard](https://tensorboard.dev/).
* Adding embeddings to the new collection in the Qdrant database.
* Updating constants in the code.

### Model Training Module

Model training module utilizes a concept of Mertic/Distance learning. Metric/Distance Learning aims to learn data 
embeddings/feature vectors in a way that the distance in the embedded space preserves the objects’ 
similarity - similar objects get close and dissimilar objects get far away. To train the model we use 
the [Pytorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) package which consists of 
9 modules compatible with PyTorch models.

The target of a model training module is to translate images into vectors in the embedding space.
Model training can be performed after following preparation steps has been completed:
* Contents of the `dataset_name` dataset has been added to the `data/metric_datasets/dataset_name` directory. 
* A `meta_dataset_name.csv` metadata file has been prepared (normally stored under `data/qdrant_storage` directory. This file contains information on the contents of the `dataset_name` dataset split by columns:
  * `` - first, empty name column, contains index number. 
  * `file` - required - name of the file. 
  * `class` - required - name of the class a given image is a part of. 
  * `label` - required - an integer representing the class. 
  * `additional_col_name` - not required - additional column with information used  for captioning images in the final application. There may be multiple columns like that one added.
* Optional training parameters (added in terminal command):
  * `data_dir` - Path for data dir.
  * `meta` - Path for meta file of dataset.
  * `name` - Name of training, used to create logs, models directories.
  * `trunk_model` - Name of pretrained model from torchvision.
  * `embedder_layers` - Layer vector.
  * `split` - Train/test split factor.
  * `batch_size` - Batch size for training.
  * `epochs` - Number of epochs in training.
  * `lr` - Default learning rate.
  * `weight_decay` - Weight decay for learning rate.
  * `sampler_m` - Number of samples per class.
  * `input_size` - Input size (width and height) used for resizing.

To run the training module run the following command in the terminal (adjust based on above list of parameters).
```
python metrics/train.py --data_dir "data/metric_datasets/dataset_name" --meta "data/qdrant_storage/meta_dataset_name.csv" --name "metric_dataset_name"
```

After training, follow steps:
* Copy `trunk.pth` and `embedder.pth` files to the `data/models/dataset_name` folder.

### Training Results

When a model training is finalized, a folder containing training results for this experiment is created. 
Part of these results showcase model performance and information regarding metrics and their evolution in 
time. To interpret these results better, this data can be ingested by 
[Tensorboard](https://www.tensorflow.org/tensorboard), providing user with necessary dashboards.

Metric logs generated during the training period can be uploaded to the [Tensorboard-dev](https://tensorboard.dev), 
which is a server-based repository of experimental results, using following command.
```
tensorboard dev upload --logdir metric_dataset_name/training_logs \
    --name "dataset_name training experiments" \
    --description "Metrics for training experiments on dataset_name dataset."
```

This command outputs a link to the dashboard containing metric charts divided by experiments.

Currently available boards:
* [Dogs](https://tensorboard.dev/experiment/oaXJiP3FRm2w9BHXzdmnJQ/#scalars)
* [Shoes](https://tensorboard.dev/experiment/DhrhyBFaSJiGwEfP9KoGDA/#scalars)
* [Celebrities](https://tensorboard.dev/experiment/TDgjYNDUQ32DoKYEZoIoow/#scalars)
* [Logos](https://tensorboard.dev/experiment/1r2BsdkER3yRBdqBjF5y9A/#scalars)
* [Waste](https://tensorboard.dev/experiment/414gt1LPSeqpAEAU1yWeIg/#scalars)


### Qdrant Database Update

Once the model is trained, a corresponding embeddings collection has to be uploaded to the Qdrant database.
It can be performed by completing the following steps:
* Modify `MetricCollections` class with a new entry for `dataset_name`.
* Add relevant reference in the `CATEGORY_DESCR` parameter.
* Copy notebook `notebooks/demo-qdrant.ipynb` to the main `visual-similarity-search` directory and run it in Jupyter.
* Run docker container containing Qdrant database.
* Run commands for (re)creating and upserting `dataset_name` embeddings to the new collection - collection name has to be the same as `dataset_name`.

Optionally collections that are not used can be deleted from the Qdrant database. 
If the Qdrant database is not based on the volume, after recreating Docker container the database will not retain inputted entries.


## Using Jupyter Notebooks

Jupyter notebooks serve as a support during the development:
* `demo-api.ipynb` - used for testing functions used by the application module.
* `demo-data-upload.ipynb` - used for uploading new datasets and related models to the MinIO storage.
* `demo-minio.ipynb` - used for testing functions of S3 MinIO data storage.
* `demo-qdrant.ipynb` - used for adding vector collections to the Qdrant storage.


## Installation Dependencies and Other Issues

* For installation on Windows, install [wsl](https://learn.microsoft.com/en-us/windows/wsl/install), [modify Docker](https://stackoverflow.com/questions/61592709/docker-not-running-on-ubuntu-wsl-due-to-error-cannot-connect-to-the-docker-daemo) and follow instructions for Linux.
* Installation dependencies are resolved and then defined by [poetry](https://pypi.org/project/poetry/). If some dependencies cannot be resolved automatically, down-/up-grading a version of the problematic library defined in the ***pyproject.toml*** file may be needed.
* According to the Docker Image's documentation, Qdrant database works on the Linux/AMD64 Os/Architecture.
* ***faiss-cpu*** library is used instead of ***faiss*** due to the former being implemented for Python's version <=3.7 only.
* A fixed version of Qdrant (v0.10.3) is being used due to its fast development and storage's versioning. Not only is a library being versioned, but collection structure does too. In consequence a collection built on Qdrant version 0.9.X is unreadable by version 0.10.X.
* On first run of the Streamlit application, when running `Find Similar Images` button for the first time, models are being loaded to the library. This is a one time event and will not influence a performance for future searches.


## Communication

* If you found a bug, open an issue.
* If you have a feature request, open an issue.
* If you want to contribute, submit a pull request.


## Authors
* [Szymon Idziniak](https://www.linkedin.com/in/szymon-idziniak-05bb01181/)
* [Bartosz Mielczarek](https://www.linkedin.com/in/bartosz-mielczarek-647346117/)
* [Krzysztof Sopyła](https://www.linkedin.com/in/krzysztofsopyla)


Want to talk about [Machine Learning Services](https://www.stxnext.com/services/machine-learning/?utm_source=github&utm_medium=mlde&utm_campaign=visearch-demo) visit our webpage. 


## Licenses

Code:
* Open-source license.

Data:
* [Shoes dataset](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k) - citations:
  * A. Yu and K. Grauman. "Fine-Grained Visual Comparisons with Local Learning". In CVPR, 2014.
  * A. Yu and K. Grauman. "Semantic Jitter: Dense Supervision for Visual Comparisons via Synthetic Images". In ICCV, 2017.
* [Dogs dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) - citations:
  * Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.
  * J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.
* [Celebrities dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) - citations:
  * Rasmus Rothe and Radu Timofte and Luc Van Gool, Deep expectation of real and apparent age from a single image without facial landmarks, International Journal of Computer Vision, 2018.
  * Rasmus Rothe and Radu Timofte and Luc Van Gool, Deep EXpectation of apparent age from a single image, IEEE International Conference on Computer Vision Workshops, 2015.
* [Logos dataset](https://www.kaggle.com/datasets/lyly99/logodet3k) - public.
* [Waste dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) - [license](https://opendatacommons.org/licenses/odbl/1-0/)

Let us know if you want your dataset removed.

# Visual Similarity Search - Category-based Image Comparison

Similarity search engine for comparing images within selected categories. 
There are two modules created within the engine:
1. Interactive Application - used for finding the closest match of uploaded or selected image within a given data category.
2. Model Training/Deployment Module - used when a new data category is added to the application.

Stable public version: [Visual Similarity Search App](https://visual-search.stxnext.pl/).


## Table of Contents
  - [Installation](#installation)
    - [Installation - Local](#installation-local)
    - [Installation - Cloud](#installation-cloud)
    - [Installation - Accessing MinIO](#installation-accessing-minio)
    - [Installation - Docker Compose Structure](#installation-docker-compose-structure)
  - [Datasets](#datasets)
    - [Current Datasets](#datasets-current)
    - [Queued Datasets](#datasets-queued)
  - [Application Module](#application-module)
  - [Model Training Module](#model-training-module)
  - [MLOps](#mlops)
  - [Using Jupyter Notebooks](#using-jupyter-notebooks)
  - [Installation Dependencies and Other Issues](#installation-dependencies-and-other-issues)
  - [Authors](#Authors)
  - [Licences](#licences)


## Installation

Both modules mentioned in the introduction use libraries specified in the ***poetry.lock*** file which are resolved
based on the contents of ***pyproject.toml*** file. To install project's libraries in your Python environment, 
install [poetry](https://pypi.org/project/poetry/) library, navigate to the directory with ***pyproject.toml*** 
in terminal and run:
```
poetry install
```

Installation and functioning of the modules depends on the `data` folder and two environment files - first for Docker-Compose build, 
and second for working of Python app.

Environment variables file for Docker-Compose is ***.env***. It contains a selection of variables:
  * `QDRANT_PORT` - port for Qdrant service,
  * `INTERACTIVE_PORT` - port for Streamlit service,
  * `QDRANT_VERSION` - version of Qdrant's docker image,
  * `PYTHON_VERSION` - used Python version,
  * `QDRANT_VOLUME_DIR` - container's volume directory for Qdrant's storage,
  * `MODEL_VOLUME_DIR` - container's volume directory for local pull of models from cloud storage,

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
├── models
│   ├── dogs
│   └── shoes
└── qdrant_storage
    ├── aliases
    ├── collections
    │   ├── dogs
    │   └── shoes
    └── collections_meta_wal
interactive
metrics
notebooks
scripts

```
This structure is split as follows:
  * `metric_datasets` - split into folders corresponding with data categories, each containing raw pictures that were used for model training and are being pulled as a result of visual search.
  * `models` - split into folders corresponding with data categories, each containing pretrained deep learning models,
  * `qdrant_storage` - storage for vector search engine (Qdrant), each data category has its own collection.

### Local

Installation using the terminal window:
* Install ***git***, ***docker***, ***docker-compose*** and ***make*** packages.
* `cd` to your target directory.
* Clone [repository](https://github.com/qdrant/qdrant_demo.git) (preferably use SSH cloning).
* Download `data.zip` file from the [Google Drive](https://drive.google.com/file/d/1UeHYtgyewXmhDd3Qd-b2ud2u65a_kc3i/view?usp=sharing) and unpack it to the repository so that the folder structure above is retained.
* To install new environment execute one of the options below.
``` 
# Use Makefile:
make run-local-build 

# Optional:
make run-local-build-qdrant-restart
make run-local-build-interactive-restart
```
* Access the visual similarity search engine under URL: [localhost](http://0.0.0.0:8080).

### Cloud

Installation using the terminal window:
* Install ***git***, ***docker***, ***docker-compose*** and ***make*** packages.
* `cd` to your target directory.
* Clone [repository](https://github.com/qdrant/qdrant_demo.git) (preferably use SSH cloning).
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
TBD 


## Datasets

Both Model Training and Application modules use the same scope of datasets. 
Each category corresponds with a single dataset.
Models are trained separately for each category. 
Application returns search results from the scope of images available only within the selected data category.

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

### Queued Datasets

* List of datasets that are queued for implementation:
  * [Logos dataset](https://www.kaggle.com/datasets/lyly99/logodet3k)
    * The largest logo detection dataset with full annotation, which has 3,000 logo categories, about 200,000 manually annotated logo objects and 158,652 images.
  * [Fashion dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
    * Thr growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon. In addition to professionally shot high resolution product images, we also have multiple label attributes describing the product which was manually entered while cataloging. 

    
## Application Module
TBD


## Model Training Module
TBD


## MLOps
TBD


## Using Jupyter Notebooks

Jupyter notebooks serve as a support during the development:
* `demo-api.ipynb` - used for testing functions used by the application module.
* `demo-data-upload.ipynb` - used for uploading new datasets and related models to the MinIO storage.
* `demo-minio.ipynb` - used for testing functions of S3 MinIO data storage.
* `demo-qdrant.ipynb` - used for adding vector collections to the Qdrant storage.


## Installation Dependencies and Other Issues

* Installation dependencies are resolved and then defined by [poetry](https://pypi.org/project/poetry/). If some dependencies cannot be resolved automatically, down-/up-grading a version of the problematic library defined in the ***pyproject.toml*** file may be needed.
* According to the Docker Image's documentation, Qdrant database works only on the Linux/AMD64 Os/Architecture.
* ***faiss-cpu*** library is used instead of ***faiss*** due to the former being implemented for Python's version <=3.7 only.
* A fixed version of Qdrant (v0.10.3) is being used due to its fast development and storage's versioning. Not only is a library being versioned, but collection structure does too. In consequence a collection built on Qdrant version 0.9.X is unreadable by version 0.10.X.
* When running the code locally, outside docker-compose build an issue with Python modules visibility may occur. In that scenario set up you terminal directory to the `visual-similarity-search` folder and run following commands in your terminal.
```
export PYTHONPATH="${PYTHONPATH}:/"
export $(grep -v '^#' .env | xargs)
export $(grep -v '^#' .env-local | xargs)
```
* On first run of the Streamlit application, when running `Find Similar Images` button for the first time, models are being loaded to the library. This is one time event and will not influence a performance for next searches.


## Communication

* If you found a bug, open an issue.
* If you have a feature request, open an issue.
* If you want to contribute, submit a pull request.


## Authors
* [Szymon Idziniak](https://www.linkedin.com/in/szymon-idziniak-05bb01181/)
* [Bartosz Mielczarek](https://www.linkedin.com/in/bartosz-mielczarek-647346117/)


## Licenses
All source code is licensed under the MIT License.

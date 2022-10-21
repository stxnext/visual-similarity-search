# Visual Similarity Search


Visual similarity search engine demo with use of [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) and [Qdrant Neural Search Engine](https://qdrant.tech/)


### Current and expected datasets
* Current:
  * Shoes dataset
  * Dogs dataset


### Training new models used in the app


### Accessing and uploading/downloading data from MinIO cloud storage


### Installing the visual similarity search engine

* Install ***git***, ***docker***, ***docker-compose*** and ***make*** package on the machine.
* `cd` to your target directory.
* Clone [repository](https://github.com/qdrant/qdrant_demo.git).
* Create **.env** file by copying the template.
* Fill parameters of **.env** file - contact [Bartosz Mielczarek](bartosz.mielczarek@stxnext.pl) for MinIO secret keys.
  * `QDRANT_HOST` - host for Qdrant service,
  * `QDRANT_PORT` - port for Qdrant service,
  * `API_HOST` - host for visual similarity search engine,
  * `API_PORT` - port for visual similarity search engine,
  * `MINIO_HOST` - host for MinIO S3 cloud storage,
  * `MINIO_ACCESS_KEY` - access key for MinIO S3 cloud storage,
  * `MINIO_SECRET_KEY` - secret key for MinIO S3 cloud storage,
  * `MINIO_BUCKET_NAME` - default bucket name in MinIO S3 cloud storage,
  * `MINIO_MAIN_PATH` - MinIO object path to input data used in similarity search engine,
  * `QDRANT_VERSION` - version of Qdrant's docker image,
  * `QDRANT_PRELOAD_APP_NAME` - name for the docker container preloading vector database in Qdrant,
  * `QDRANT_APP_NAME` - name for the Qdrant database docker container,
  * `INTERACTIVE_APP_NAME` - name for the visual similarity search docker container,
  * `SHARED_VOLUME_DIR` - shared volume path, where data input for Qdrant is being preloaded from MinIO storage,
  * `TYPE` - environment type.
* Run `docker-compose up -d --no-deps --build`.
* After `QDRANT_PRELOAD_APP_NAME` ends run `docker compose restart QDRANT_APP_NAME` - this will reload pulled data to the Qdrant database.
* Run `docker compose restart INTERACTIVE_APP_NAME` - this will reload respectable models and reestablish connection to th `QDRANT_APP_NAME`.
* Access the visual similarity search engine under URL: http://0.0.0.0:8080.

### Accessing the visual similarity search engine on STX Next server


### Using Jupyter Notebooks


### Communication

* If you found a bug, open an issue.
* If you have a feature request, open an issue.
* If you want to contribute, submit a pull request.


### Licenses
All source code is licensed under the MIT License.

A containerized model inference server deployment to predict house price from numeric inputs.

The local development env is presumed to use conda. You can install a minimal version of conda with miniconda [here](https://www.anaconda.com/docs/getting-started/miniconda/install). You can activate the `housing` env with conda by running 
```sh
conda env create -f conda_environment.yml
conda activate housing
```
Any updates to the conda env for development should be reflected in `conda_environment.yml`

`rest_api.py` spins up the model prediction server and is intended to be run Dockerized. You should [install](https://www.docker.com/) Docker if you don't have it. First ensure you have ran `python create_model.py` to generate the necessary model outputs. You can then build and run the Docker image to spin up the web server locally via
```sh
docker build -t housing-api .
docker run -p 8000:8000 housing-api
``` 

Note that the Docker image uses pip rather than conda for package management. This is to keep the containers as lightweight as possible. To achieve this without much added complexity, we maintain a `requirements.txt`, containing package dependencies in a format ingestible by pip. If adding package dependencies for `rest_api.py`, be sure to update `requirements.txt` with the same package version as used by your local conda env.

Once you have the housing server running in Docker, you can test it out by running `python test_rest_api.py` which will invoke the prediction endpoint for each entry of `data/future_unseen_examples.csv`.

We also have a model evaluation script you can run to get an idea of model fit and variance: `python evaluate_model.py` whose output has been saved to `evaluate_model_output.txt`.

We've added a script to test an updated XGBoost model `create_model_improved.py`. We output Shapley values for the model under `shap_feature_importance.csv`. These indicate rather intuitively that sqft_living has high predictive value for the housing price. The results may be used for further model adjustments such as recursive feature elimination.

For closer real world demonstration we also implement a deployable lightweight local NGINX reverse proxy setup, routing traffic to two backend server containers, configured via `docker-compose.yml` and `nginx.conf`. To run this locally:
```sh
docker-compose up -d
docker-compose ps
```
You can run `python test_rest_api.py` the same as before with this deployment. To view the logs for the various services we spun up, you can run `docker-compose logs -f [nginx/app1/app2]`. To tear down the deployment, run `docker-compose down`.

This code is meant to be extended into a Kubernetes deployment for any real world usage. That entails autoscaling of model server containers based on configured pod metrics such as CPU usage and queue depth. Kubernetes can spin up an NGINX Ingress gateway for accepting external traffic, validating, and forwarding requests to the containers through a configured load balancer. Models are bundled with the server image in this implementation, so any model updates only require a rolling restart of the deployed containers which Kubernetes can handle, versus the application server itself dispatching to a model server to fetch the latest model, or keeping manual track of whether a loaded model is stale relative to an external source.

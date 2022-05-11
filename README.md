# Introduction

This project was completed as part of the requirements for the Udacity MLOps Nanodegree. Below is a breakdown of the folders and files:

- `data`: this folder contains the raw and cleaned data required for the completion of the project. To populate the folder with the data you will need to run `dvc pull`
- `model`: this folder contains the model card (i.e. `Model_card.md`), and the pickles required to run the api. The pickles can be downloaded by running `dvc pull`
-  `screenshots`: this folder contains the screenshots required for the completion of the project
- `starter`: this folder contains the source code for the training, evaluation, and saving of the model related files
- `tests`: this folder contains the `pytest` modules as well as the live api test script.
- `main.py`: this file contains the `FastAPI` implementation of the `API`
- `requirements.txt`: this file contains the dependencies required to install and run this project
- `Procfile`: this file contains the `Heroku` job definition
- `Aptfile`: this file contains the `dvc` dependency to be installed inside the `Heroku Dyno`
- `.dvc`: contains the `dvc` configurations for this project
- `.github/workflows`: contains the `github actions` CI workflow for this project

#!/bin/bash

ENV_NAME=fakenews
deactivate
rm -rf $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install --upgrade pip 
pip install ipykernel
pip install transformers matplotlib 
pip install pandas
pip install wordcloud
pip install scikit-learn-intelex
pip install nltk
pip install seaborn tqdm
python -m ipykernel install --user --name $ENV_NAME  #Register the env as a kernal for using it with jupyter notebook

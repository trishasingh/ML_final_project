# Instructions for running

    to install requirements call
        > pip install -r requirements.txt
## To train:

    > python3 data_parse.py

    The -s option skips the creation of the data file if included (need to run at least once without -s initially)
    the -n option says to skip training
    the -g option is for when training is going to happen on a gpu
    the -m option specifies a model to load and continue training if one is availible

## to forecast/evaluate

    > python3 machine_learn.py -m path/to/model

    the -m option specifies which model to evaluate
    can include -n option to skip forecasting and just graph the predictions from the nueral net
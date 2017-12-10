# Instructions for running

    To install requirements call
        > pip install -r requirements.txt

Just a warning, all of these can be very slow to run (especially without a GPU).

## To train:

    > python3 data_parse.py

    The -s option skips the creation of the data file if included (need to run at least once without -s initially)
    the -n option says to skip training
    the -g option is for when training is going to happen on a gpu
    the -m option specifies a model to load and continue training if one is availible

## To forecast/evaluate:

    > python3 machine_learn.py -m path/to/model

    the -m option specifies which model to evaluate
    can include -n option to skip forecasting and just graph the predictions from the neural net

This script will generate a plot at the end that shows how well it did in predicting and forecasting for the set given.
the plot is interactive and one can zoom in on any areas of interest.

## To evaluate battery sizes:

    > python3 battery.py
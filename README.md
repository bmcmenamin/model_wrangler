# model_wrangler

## warning
This has minimal testing and is still a work in progress

## Rationale
I have a number of side-projects using tensorflow and/or keras, and I found myself constantly re-implementing the same pieces of boring model-wrangling code -- saving/restoring weights, dividing datasets into train/test samples, etc. So I decided to make a general package or reusable network pieces that'd automate the construction of a SkLearn-like API around my tensorflow/keras models.


The package in `./model_wrangler` handles all that high-level model stuff. The folder `./model_wrangler/corral` is where you store the configs for specific models. I'll write better documentation later. Speaking of, here's my partial to-do list:

* Implement models from side projects into this format
    * DataManager for sequential/Timeseries data
    * Compatability with siamese/triplet training

* Documentation
    * Set up toy problems as demos in a notebook
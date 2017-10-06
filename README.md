# model_wrangler

## warning
This has minimal testing and is still a work in progress

## Rationale
I have a number of side-projects using tensorflow and/or keras, and I found myself constantly re-implementing the same pieces of boring model-wrangling code -- saving/restoring weights, dividing datasets into train/test samples, etc. So I decided to make a general package or reusable network pieces.

I'm also a big fan of Keras for simplifying complex TensorFlow operations, but I wanted to be able to work in pure tensorflow so I could start building unconventional networks that may not be easy to do with Keras.

The package in `./model_wrangler` handles all that high-level model stuff. The folder `./model_wrangler/corral` is where you store the configs for specific models. I'll write better documentation later. Speaking of, here's my partial to-do list:
* Implement models from side projects into this format
    * DataManager for sequential/Timeseries data
    * Compatability with siamese/triplet training

* Documentation
    * Set up toy problems as demos in a notebook
* Unit tests

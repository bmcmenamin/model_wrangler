# model_wrangler

## Rationale
I build a lot of models with tensorflow and/or keras in various side projects. Unfortunately, I end up spending most of my time writing the 'boring' parts of the models -- functions to save/restore weights, handling how we divide datasets into batches, etc. So I've made this repo which will prevent me from re-writing all the same boring chunks of code for model-wrangling and wrap my TensorFlow models in a SkLearn-esque API to simplify common operations.

The package in `./model_wrangler` has the base classes that handle most of the common model operations, and `./model_wrangler/corral` has the specific model implementations you'd actually use.

There's more documentation in `./model_wrangler/corral` and `./model_wrangler/tests`.

Here's my partial to-do list:

* Add more tensorboard scalars during training
* Add functionality for timeseries data:
    * recurrent models (e.g., LSTM)
    * DataManager for sequential/Timeseries data
* Clean up logging
* Expand documentation

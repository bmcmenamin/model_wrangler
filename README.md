# model_wrangler

## Rationale
I build a lot of models with tensorflow and/or keras in various side projects. Unfortunately, I end up spending most of my time writing the 'boring' parts of the models -- functions to save/restore weights, handling how we divide datasets into batches, etc. So I've made this repo which will prevent me from re-writing all the same boring chunks of code for model-wrangling and wrap my TensorFlow models in a SkLearn-esque API to simplify common operations. I could probably find a way to use TensorFlows built in estimators and dataset tools, but I'm learning more doing it this way.

The `ModelWrangler` class in `./model_wrangler.py` does most of the orchestration. It takes a a dataset manager from `dataset_managers.py` and a TensorFlow network architecture derived from `architecture.py` and sets up a bunch of useful model operations. The directory `./model_wrangler/model/corral` has a bunch of pre-made model architectures you may want to use.

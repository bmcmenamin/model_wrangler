# modelwrangler/corral

This is the model corral where a bunch of premade models live.

Each of the .py files here defines a class for a type of model (e.g., Linear Regression).
It should contain a `<model_name>Model` class that is derived from `BaseArchitecture` and overwrites the method `setup_layers` to build this particular model architecture using whatever dicts of params were passed in to the `__init__` call.

The `setup_layers` method should return the following tensorflow objects:
* list of model input layers (`in_layers`)
* list of model output layers (`out_layers`)
* list of placeholders used as target output values during training (`target_layers`)
* optional list of model 'embedding' layers (`embeds`)
* a cost function that is minimized during training (`loss`)


* A class named `<model_name>` that inherits the `ModelWrangler` class and has an __init__ method that calls the init method in `<model_name>Model`. Now you can import `<model_name>` from this python model and you'll automatically get usefull functions like:
    * initialize
    * train
    * predict
    * embed
    * score
    * feature_importance
    * get_from_model (get activations/weights from a layer by name)
    * save
    * load


Current models:
* `linear_regression`: Linear Regression
* `logistic_regression`: Logistic Regression

* `dense_autoencoder`: Is an autoencoder built with densely connected layers
* `convolutional_autoencoder`: Is an autoencoder built with convolutional layers

* `dense_feedforward`: Feedforward net with all dense connections model
* `convolutional_feedforward`: Feedforward net with convolutional layers and then dense layers
* `convolutional_siamese`: Convolutional networks trained with siamese pairs

* `text_classification`: A convoluional feedforward net that takes strings as inputs and does all the conversion to numerics internally
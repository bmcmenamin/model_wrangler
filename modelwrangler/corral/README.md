# modelwrangler/corral

This is the model corral where a bunch of premade models live.

Each of the .py files here defines a class for a type of model (e.g., Linear Regression).
It should contain the following pieces:

* A `<model_name>Params` class that inherits from `BaseNetworkParams`. This class needs two dictionaries as class attributes:
    * Set `MODEL_SPECIFIC_ATTRIBUTES` to define default parameters for everything in your model
    * Set `LAYER_PARAM_TYPES` to indicate which of the parameters in MODEL_SPECIFIC_ATTRIBUTES need to be cast from dict to special layer config objects (e.g., `LayerConfig`)

* A `<model_name>Model` that inherits from `BaseModel`.
    * Set the class attribute `PARAM_CLASS` equal to the class used for default parameters. Usually it'll be the `<model_name>Params` you've set up in the previous step.

    * Define a method named `setup_layers` that will set up all the tensorflow layers that make up your model. It should return the following tensorflow objects: model input layer (`in_layer`), model output layer (`out_layer`), placeholder used to hold target values during training (`target_layer`), and a cost function that is minimized during training (`loss`). The `BaseModel` gives you a bunch of methods that are helpful for this, like `make_dense_layer` and `make_conv_layer`. 

* A class named `<model_name>` that inherits the `ModelWrangler` class and has an __init__ method that calls the init method in `<model_name>Model`. Now you can import `<model_name>` from this python model and you'll automatically get usefull functions like:
    * initialize
    * train
    * predict
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
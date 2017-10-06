# model_wrangler/corral

This is the model corral where a bunch of premade models live.


Each of the py files here defines a class for a type of model (e.g., Linear Regression) that has the guts of TensorFlow in it. That class gets passed into the ModelWrangler class, and it is suddenly give the powers of being able to save/load, batch train, predict, measure feature importance, etc.

Current models:
* `linear_regression`: Linear Regression
* `logistic_regression`: Logistic Regression

Future models:
* autoencoders
* conv nets
* and more. stay tuned.

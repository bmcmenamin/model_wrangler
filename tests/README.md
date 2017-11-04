# model_wrangler Tests

When I started out, I had envisioned writing a tonne of unit tests. But after a few days, I realized that was boring and I'd probably end up dropping the whole project if I had to spend my free time doing that.

Instead, I've done two things:

* Based on this [great blog post](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) by Chase Roberts about unit testing for machine learning, I've come up with a couple baseline unit tests that should apply across all model types and have set them up as methods in the `ModelTester` class. Initialize the ModelTester by passing in a model class (e.g., 'LinearRegression'), and it'll automatically run those tests on it.

* I set up crude end-to-end tests for troubleshooting each new type of model. These generally initialize a model and run a couple epochs of training on it to make sure that the cost function starts going down.

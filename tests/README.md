# model_wrangler Tests

When I started out, I had envisioned writing a tonne of unittests. But after a few days, I realized that that sounded boring and I'd probably end up dropping the whole project if I had to spend my free time doing unit tests.

So I started to rely on crude end-to-end tests for troubleshooting each new type of model. However, I recently came across this [great blog post](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) by Chase Roberts about unittesting for machine learning and have decided to standardize a few of these basic unit-tests that can be applied across models and store them in the ModelTester object defined in `tester.py`.

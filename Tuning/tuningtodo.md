What do we want to do for the hyperparam tuning?

* Basically, algorithm should define a function that defines the model using the given parameters, trains it and evaluates it, spitting out the mean of the evaluations as the parameter to optimise over
* Could parallelise the training part, but seems far far far more useful to parallelise the evaluation part (that takes forever!)

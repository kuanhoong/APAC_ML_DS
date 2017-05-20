############################################################
## APAC Machine Learning & Data Science Community Summit  ##
## Seoul, Korea                                           ##
## 20th May 2017                                          ##
## Malaysia R User Group                                  ##
## Poo Kuan Hoong, Ph.D                                   ##
############################################################

## h2o Deep Water Demo

library(h2o)
h2o.init(nthreads=-1)
if (!h2o.deepwater.available()) return()

iris <- iris
train <- as.h2o(iris)
predictors=1:4
response_col=5
hidden_opts <- list(c(20, 20), c(50, 50, 50), c(200,200), c(50,50,50,50,50))
activation_opts <- c("tanh", "rectifier")
learnrate_opts <- seq(1e-3, 1e-2, 1e-3)
max_models <- 1000      ## max number of models is 1000
nfolds <- 3             ## use cross-validation to rank models
                        ## and to find optimal number of epochs for each model
seed <- 42
max_runtime_secs <- 30  ## limit overall time (this triggers)

# Set the hyperparameters
hyper_params <- list(activation = activation_opts,
                     hidden = hidden_opts,
                     learning_rate = learnrate_opts)

search_criteria = list(strategy = "RandomDiscrete",
                       max_models = max_models,
                       seed = seed,
                       max_runtime_secs = max_runtime_secs,
                       stopping_rounds=5, ## early stopping of the overall leaderboard
                       stopping_metric="logloss",
                       stopping_tolerance=1e-4)

# setup the deep water grid
dw_grid = h2o.grid("deepwater",
                   grid_id="deepwater_grid",
                   x=predictors,
                   y=response_col,
                   training_frame=train,
                   epochs=500,        ## long enough to allow early stopping 
                   nfolds=nfolds,              
                   stopping_rounds=3, ## enable early stopping of each model in the hyperparameter search
                   stopping_metric="logloss",
                   stopping_tolerance=1e-3,    ## stop once validation logloss of the cv models doesn't improve enough
                   hyper_params=hyper_params,  
                   search_criteria = search_criteria)
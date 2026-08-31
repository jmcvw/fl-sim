# "Federated Learning simulation"


============================================================

Simple Federated Learning simulation
Bootstrap + rolling window + gradient descent

============================================================

## Model  

Petal.Length ~ Petal.Width

Three organizations; each organisation:  

- has its own bootstrap sample of data (75 rows)
- receives 15 new observations per round
- retains only the most recent 75 observations
- this simulates streamed data
- starts local training from the previous round's starting parameters
- performs a limited number (default 20) of gradient-descent iterations
- sends its local parameters to the aggregator
- new aggregated parameters become the starting parameters for the next round

The aggregator:  

- averages the local parameters
- produces new global parameters


The new global parameters become available for the NEXT round.
They are not used to alter the completed round retrospectively.



## GLOBAL PARAMETERS

At the beginning there is no trained model, so we start
with arbitrary parameters: intercept = 3; slope = 0

After each round, these are updated with the newly aggregated global parameters



## Create each organisation's local data

Each organisation independently bootstraps 75 rows from iris, ___with replacement__.  
Each organisation receives 15 new observations per round.

## Evaluate the previous global parameters

This tells us how the parameters received at the beginning
of the round perform on each organisation's local window.



## Aggregation

The organisations send their __local parameters__
They do __not__ send their data.

For simplicity we use an unweighted mean.


## Local training

Every organisation starts from the __same__ global parameters.

But each organisation performs the optimisation against
its __own__ local data.



## End of round

Aggregated_params are the result of __the current__ round.

They become the global parameters available at the __start of the next__ round.

The simulation is set up to iterate over 4 rounds

The next round will

- receive 15 additional observations
- move the rolling window (drop first 15 obs)
- start from the previous global parameters
- perform local gradient descent
- compare RMSE before/after
- aggregate the local parameters

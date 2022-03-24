# Final Model

This repository contains the final model which has only 2.63 million parameters and can achieve up to 91.12% accuracy. What we did was:
1. Use Adam Optimizer
2. Augment data and combine with original training dataset
3. Set number of blocks to 2 of sizes 3 and 10 respectively
4. Set c1=58. Subsequent values of ci are calculated by doubling
5. Avg pool kernel size- 2
6. Cross validation to narrow down highest accuracy touched
7. Run for for 100 epochs

We tried other things like L2 regularization, Gradient Clipping, raytune etc. spanning a total of 70-80 different models and the above config gave best results


## TEAM -  Vara Lakshmi Bayanagari, Sayan Banerjee, Aditya Chawla

# Bias-Variance tradeoff
## Setup
* In general machine learning the search for an "optimal" predictor within a class of models $\mathcal{H}$ is done via some optmization method
* The "size" of $\mathcal{H}$ is considered key to find a good model. To small and the model is expected to underfit and too large and the model will fit the training data perfreclty while generalize poorly
* In deeplearing we deal with very large, over-parameterized models (#free parameters >> data points), however, even though they can fit the  training data at a near zero loss, such solutions seem to also generalize well to unseen data which is not what would be expect from conventional wisdom

## Questions
* Why can overparameteriized models generalize? 
* Why does SGD lead to solutions which seem to be global minima in a high-dimensional non-convex "landscape"?
* What is the connection between 

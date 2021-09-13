# xai_tracking

This is the pytorch implementation for our paper "Explaining Deep Learning Representations by Tracing the Training Process".

> __Abstract__
> We propose a novel explanation method that explains the decisions of a deep neural network by investigating how the intermediate representations at each layer of the deep network were refined during the training process.
> This way we can a) find the most influential training examples during training and b) analyze which classes attributed most to the final representation.
> Our method is general: it can be wrapped around any iterative optimization procedure and covers a variety of neural network architectures, including feed-forward networks and convolutional neural networks. We first propose a method for stochastic training with single training instances, but continue to also derive a variant for the common mini-batch training.
> In experimental evaluations, we show that our method identifies highly representative training instances that can be used as an explanation. Additionally, we propose a visualization that provides explanations in the form of aggregated statistics over the whole training process.

Currently contains two experiments: cifar10 and gnn that can be executed via `python -m {cifar10|gnn}.train` to train the model and `python -m {cifar10|gnn}.explain` to generate a sample explanation.

__To-Do:__

- [ ] restructure into proper python package
- [ ] include new experiments
- [ ] push approximated versions for less storage demand

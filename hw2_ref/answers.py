r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    lr = 0.02
    wstd = 0.5
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.5
    lr_vanilla = 0.02
    lr_momentum = 0.0045
    lr_rmsprop = 0.00025
    reg = 0.0
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.5
    lr = 0.001 #0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1 + 2. Yes they match the results we expected. Dropout is used as regularization to help prevent overfitting and generalize better. As we can see when the dropout is 0 train acc is about 65% and test acc is about 21%, on the other hand with dropout=0.4 we get worse training acc of 39% but better test acc with 25.5%. Also when looking at the graph of the test acc for dropout=0 we can see it starts to go down, indicating of overfitting, in contrast the graph when dropout=0.4 continues to rise, which indicates that the dropout actually helps at preventing overfitting.
With dropout=0.8 we get worse results, because dropping too much neurons at training time might actually hurt the network's complexity (also according to http://papers.nips.cc/paper/4878-understanding-dropout.pdf we get the best regulariztion with dropout around 0.5, which matches our results), maybe when running with more data and epochs we will be able to get better results (because high dropout means we train less neurons each epoch).
Overall the low dropout settings performed much better than the high dropout settings because what we mentioned above, low to moderate dropout should mostly give better results than very low or high dropout.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
That might happend because of the softmax function, meaning we might get high accuracy because the softmax selects the correct class, but other in-correct classes still get high scores (but not the maximum), so if we get better accuracy on test (due to actual improvement) but we get higher scores for the in-correct classes our loss might get worse (due to the cross-entropy loss).

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
We can see that at best we got around 90% training acc, and 67% test acc, and we stop just after the overfitting point (where the loss and acc of the test set start to get worse).
We also got better results with K=64 than K=32 (less loss and more accuracy), that might happen because with more filters we have a more complex network (can learn better)
1. From the results we got, increasing the network's depth actually hurts the accuracy. We got the best results on L=2, and on L=8,18 the network was un-trainable.
   That depth seems to keep the network simple enough to be easily trained with our data (not too many layers) but still expressive enough.
2. We got that for L=8,16 the network was un-trainable. That might happend when there are too many parameters and the target function becomes too complex for our gradient-based optimization algorithm to actually get to a good minimum or we lack computational power to do so, e.g. if we use a big batch size we might get good results but it's too expensive.
Getting more data, and increasing batch sizes might help (because we got better approx of actual gradient, so the optimization algo' will work better).
Reducing the features dimension (e.g. less features to train each epoch), for example using droput.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**
Just like before, we say that with bigger L the network becomes worse (both loss-wise and accuracy-wise), and becomes un-trainable at some point.
Also bigger K got us better results as well, just like in the previous case.
Training also stops when reaching the over-fitting point.
Our best results are with L=2, K=256, we get around 90% training acc and 70% test acc (at the best point).
Generally we got similar results to the exp 1.1, but here we can notice for larger K's the network converges much faster (overfitting also starts earlier).

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**
Like the previous results, we can see that bigger L hurts the network until it becomes un-trainable.
Our best results here are with L=1, and we get around 85% training acc and 73% test acc (which is the best test acc so far).
The training went just like the other experiments.
The most noticable thing here is that although we didn't get the best *training* acc we did get the best *test* acc, which means that with varying K's the network is able to generalize better than with a single, fixed value of K.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**
1. We took our inspiration from the following article: https://arxiv.org/pdf/1603.05027.pdf (Pre-act ResNet).
   The network is generally similar to Res-Net that we saw in class.
   The network's basic building blocks (ResNetBlock) are BatchNorm->ReLU->Conv->BatchNorm->ReLU->Conv, with an identity shortcut (which is very important expirmentally according to the paper),      using batchNorm which helps alot (as we saw in class).
   The network itself looks like: Conv->ResNetBlock_1*L->ResNetBlock_2*L->ResNetBlock_3*L->ResNetBlock_4*L->AvgPool->Linear, where each ResNetBlock has different conv. sizes (according to the      given K), the first one is dimension-preserving, each ResNetBlock*L has the first ResNetBlock with a shortcut and the others without.
   We used almost all of the ways we saw in class to imrpove CNN - different strides, skipping connections and Batch Norm.
2. Again the bigger the L the worse are the results. Also the training process went exactly like the previous exps.
   Here we got our best results with L=1, we got almost 100% training acc and almost 85% test acc, which is far better than any other exps.
   In the case where L=1, we can see the loss is very small in both training and test.
   Overall we got much better results with the custom network, so planning and building it was really worth the time.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

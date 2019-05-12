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
    wstd = 0.1
    lr = 0.05
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr_vanilla = 0.02
    lr_momentum = 0.003
    lr_rmsprop = 0.00025
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
With a dropout, I expected that the accuracy on the training set will decrease and the accuracy on the test set will increase, compared
to no dropout. This was my expectation because dropout is used to help prevent overfitting. In addition I expected that the
train accuracy will decrease when the dropout value is increased,

With no dropout and with dropout=0.4 I got a result that match my expectation. However, dropout=0.8 didn't match my expectation.

With no dropout we got around 95% train accuracy (overfit) and only 24% test accuracy, and with dropout=0.4 I got around 55% train accuracy and
30% test accuracy that match our expectation (worst train accuracy for the sake of better test accuracy).  

For dropout=0.8, we can see that the train accuracy is decreased comparing to the no dropout but increased comparing
to dropout=0.4. However, we still see the the test accuracy is same or even worse comparing to the no dropout which I didn't expect.
This can be explained because dropping too much neurons at train time can hurt the training process and instead of preventing overfitting, 
we cause the network to under-learn. In addition, with larger dropout the training should last more time (more epochs) to converge.
 
 This mean that a dropout value should not be too large and shouldn't be too small, therefore 0.4-0.5 gave the best results. 

"""

part2_q2 = r"""
Yes, increasing in both the accuracy and loss is possible because loss is a continuous value average while accuracy is a discrete 
value (class) depending.

The softmax and class prediction can select the correct class while the actual probability is going far from the max but still not crossing 
the correct threshold. 
For example, 
let's take a look on a simple example with 2 classes classification. If we have two samples that should be classified to 1, 
and the threshold is 0.5 . [0-0.5] -> classified 0 and [0.5-1] -> classified 1. If at the current step sample 1 have value of 0.4
it will be classified to class 0 [incorrect] and sample 2 has value 0.9 it will be classified to class 0 [correct] wich mean
we have 50% accuracy. If after one step sample 1 have values 0.51 and sample 2 have value 0.6 both will be classified correctly to class 1
which mean 100% accuracy, however the loss has increased because in total the two samples value are now more far from the actual desired
max size 1.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. best test accuracy that we got is around 58% and around 79% for training set. 
2. better and faster converge for K=64 compared to K=32
3. Increasing the network depth (L) hurts both the accuracy and loss, and for some large values of L (starting from L =8)
   the network is not trainable at all.
   With the smallest value L=2, we got the best accuracy and loss for both K=32 and K=64. Deeper is more parameters in our network,
   I think that bigger L perform worse because the database has only 10 classes with 60,000 samples which mean that we don't
   need a lot of parameters for it. Therefore going deeper is not better.
4.For L=8 and L=16 the network failed to train. This can be explained with the huge number of parameters that need to be 
  trained causing the target function to be to complex for our optimizer to keep converge for every step. This can be solved by reducing
  the number of parameters needed to be trained every epoch by using a dropout for example, or maybe increasing the batch size.
"""

part3_q2 = r"""
The most obvious thing from looking at this experiment graphs, the larger the num of filters in the convolution layer the better
the accuracy and loss and the faster the converge. In the two experiments were L=2 and L=4 the model converged, and for L=8 the
model was un-trainable for all the values of K (same result as previous experiment). For K=258 we got the best accuracy on both 
the training set which achieved 82% accuracy and the test set which got 62% accuracy for the smaller L.

This experiment proves the result from previous experiment that smaller L is better (less deeper). What's new from this one?
Larger K give us better accuracy and faster converge. 
"""

part3_q3 = r"""
Now, not just the L affects the depth of the network sense every layer now is in depth=3. Therefor, we can see that in this
experiment that only for L=1 the network is trainable. However, different number of filters in 3 convolution layers converged
with better accuracy and loss on both train set and test set compared to L=2 and K=256 for example. This mean that varying the number 
of filters in the convolution layers help compared to fixed K. 
This experiment achieved the best accuracy for both train and test set achieving around 86% accuracy for train set and 70%
accuracy for test set. 
 
 And again, like the previous experiments, bigger L hurts the network causing it being un-trainable.
"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

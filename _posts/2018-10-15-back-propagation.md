---
layout: post
title: "Backpropagation"
published: 2018-10-15
updated: 2018-10-15
preview: "Making our neural network learn."
---

Today, we will discuss the [backpropagation algorithm](https://en.m.wikipedia.org/wiki/Backpropagation) which will allow our networks to learn. Backpropagation is probably the most  complex part of our neural network, so don't stress too much about the mathematics. Try to understand it conceptually and take note of the important formulas so you can implement it correctly. Finally, try to understand what computations are being performed at each layer, and what order they are being performed in.

For additional information on backpropagation, see these posts.
+ Video: [What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
+ Video: [Backpropagation, Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)
+ Article: [Neural Networks and Backpropagation Explained in a Simple Way](https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e)
+ Book Chapter: [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)



<hr>
## After Reading This Post
After reading this post, you should be able to answer the following questions.
1. What is a cost function?
2. What properties should a cost function have?
3. What are the backpropagation equations?
4. What outputs can we cache during forward propagation for use in backpropagation?

The answers are provided at the end.



<hr>
## What is Learning?
We have said that backpropagation is the algorithm for making our neural networks "learn," but we haven't said what exactly it means for our neural network to learn. Suppose, for now, that we're trying to create a neural network that can recognize handwritten digits and we have 1,000 training examples. By training examples, we mean examples where we have the input (the image) and the true output (what digit it is a photo of).


### Classification Error
One thing we could do is feed of all of these digits into our network and see how many it gets correct. We could then say our neural network is "learning" if the number of digits it gets correct is increasing during each iteration. Or the other way around, we could say our neural network is learning if the number of images it gets wrong is decreasing each iteration (I'm restating it this way because what we will actually do is minimize the error rather than maximize the accuracy).

This is known as the _classification error_ of our neural network, and it is a good, human-readable metric, but it is not useful for training our network. There are two issues with it. First, it is not a clear function of the weights and biases in our network. We want to measure our error as a function of our weights and biases so we have some intuition on how to change them. The second issue with this measure is that there is no concept of "closeness," our output is either 100% right or 100% wrong. But what if our neural network _almost_ predicted the right output for one of our examples? We want to be able to recognize this and adjust accordingly.


### Cost Functions
Instead, we want to find a _cost function_ with the following form.

$$C(W^{[1]}, b^{[1]}, \dots, W^{[L]}, b^{[L]}) = \frac{1}{m} \sum_{i=1}^{m} E(x^{(i)})$$

In other words, we want to find a cost function with a few properties. First, we want it to be a function of our weights an biases. Second, we want it to be an average of the errors for each training example individually. Finally, we want to be punished more for very bad predictions than for predications that are almost right. The greater the value of our cost function, the more wrong we are. If we can minimize the cost function (by altering the weights and biases), then our neural network should make better predictions. Although we aren't directly measuring the classification error, it should improve as well.


### Minimizing our Cost Function
How can we minimize such a function? If you have taken a multivariable calculus class, you may have heard about the _gradient_ of a function. The gradient of a function is a vector which tells us, "If you move each input (the weights and biases) in the direction I am pointing, you will increase the function output (the cost) mostly rapidly." Conveniently, it also tells us the following, "If you move each input (the weights and biases) in the opposite direction (multiply by -1) of where I am pointing, you will decrease the function output (the cost) mostly rapidly." I am now obligated to give you the canonical skier example. If you were a skier on a mountain, the gradient would point in the direction of steepest ascent. By skiing in the opposite direction of the gradient, you will reach the bottom of the mountain most quickly.

Long story short, we can minimize the cost function by doing the following.
1. Randomly initialize our weights and biases
2. Determine how good/bad our predictions are. (Forward Propagation + Cost Function)
3. Find the gradient of our cost function. (Backpropagation!)
4. Update our weights and biases by subtracting the gradient.
5. Repeat steps 2 - 4 until we're happy.

Of course, we wouldn't have spent all this time talking about cost functions and how to minimize them if we didn't have any. We will now look at two cost functions. For each one, observe that (1) it is an average of the error over each example, and (2) it is a function of our weights and biases.


### Example Cost Function: Mean-Squared Error
The first example is a cost function you have probably seen before. It is sometimes called the sum of squared differences or the mean-squared error (MSE) function. For each example, it measures how far it is from the desired output and adds that to our total cost. If our output perfectly matches the target output, the value will be 0. If our output is very far from the target output, the value will be very large. It has the following form.

$$C = \frac{1}{2m} \sum_{i=1}^{m} ||y^{(i)} - a^{[L](i)}||^2$$

Let's clarify some of the notation real quick
+ $$m$$ is the number of examples we fed into our network
+ $$y^{(i)}$$ is the desired (target) output of our neural network for example $$i$$
+ $$a^{[L](i)}$$ is the actual output of or neural network for example $$i$$. The $$L$$ means this is the activations from the final layer, which is the same as our output.

It should be clear that this is an average of the errors of each example, but it is a little less clear that this is a function of the weights and biases. After all, there isn't a single 'w' or 'b' in that formula. Instead, this is a function of the output of our neural network. Recall, however, that the output of our neural network _is_ a function of our weights and biases. Therefore, we can use the multivariable chain rule as follows to find the gradient with respect to the weights and biases as desired.

$$
\begin{align}
\frac{\partial C}{\partial W} & = \frac{\partial C}{\partial A} \frac{\partial A}{\partial W} \\
\frac{\partial C}{\partial b} & = \frac{\partial C}{\partial A} \frac{\partial A}{\partial b}
\end{align}
$$

If that doesn't make much sense to you just remember this: Our cost function depends on our weights and biases because it depends on the output of our neural network.


### Example Cost Function: Cross-Entropy Cost
The cross-entropy cost function is another commonly used cost function. It is less intuitive, and it comes from maximum-likelihood estimation in statistics. The cross-entropy seems to get better results then MSE, so we're going to use it in our neural network. The formula is as follows.

$$C = \frac{-1}{m} \sum_{i=1}^{m} \big[ y^{(i)} \log(a^{[L](i)}) + (1 - y^{(i)}) \log (1 - a^{[L](i)}) \big]$$

Again, it should be clear that this is an average of the errors of each example. Again, it is not quite clear that this is a function of the weights and biases. But just like the MSE function, since it depends on the output of our neural network, it also depends on the weights and biases.


### Summary and Further Reading
That was quite a bit of reading -- let's summarize. The classification error is a good, human-readable way of measuring the performance of our neural network. For training our neural networks, we will try to minimize the Mean-Squared Error, the Cross-Entropy Cost, or some other cost function. We train our neural networks by moving in the opposite direction of the gradient of the cost function. Each cost function has two important components: (1) it is an average of the errors of each example, and (2) it depends on the weights and biases of our network.

Here are three good articles on cost functions:
1. [Loss Functions in Deep Learning](http://yeephycho.github.io/2017/09/16/Loss-Functions-In-Deep-Learning/)
2. [Why You Should Use Cross-Entropy Error](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)
3. [The Cross-Entropy Cost Function](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function)



<hr>
## Backpropagation: Layer by Layer
Finally we can discuss the backpropagation algorithm. "Forward" Propagation goes forward, so it shouldn't be too hard to believe that "backpropagation" goes backward -- that is, we start at the output layer and work towards the input layer. I am basically just going to give the formulas for each layer of backpropagation, and then I will emphasize making sure it's clear what goes on at each layer. First, a few things to keep in mind.


### Before We Begin
Keep the original forward propagation formulas in mind. In forward propagation we start with $$A^{[l-1]}$$, then compute $$Z^{[l]}$$, then finally $$A^{[l]}$$. In backpropagation we will start with $$A^{[l]}$$, then work towards $$Z^{[l]}$$, then finally $$A^{[l-1]}$$. Keep these formulas in mind.

$$
\begin{align}
Z^{[l]} & = W^{[l]} A^{[l-1]} + b^{[l]} \\
A^{[l]} & = g(Z^{[l]})
\end{align}
$$

Finally, recall that our ultimate goal is to find the gradient of the cost function with respect to the weights and biases. The cost function is a function of $$A^{[L]}$$ -- the output of our final layer. The output of our final layer is a function of all our weights and biases. We can use the chain rule to get the gradient with respect to the weights and biases. It's an oversimplification, but basically just cancel out the denominator of one derivative with the numerator of the next derivative until you reach the variable you want. You can read more about the chain rule [here](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version) and [here](http://tutorial.math.lamar.edu/Classes/CalcIII/ChainRule.aspx). See what I mean below.

$$
\begin{align}
\frac{\partial C}{\partial W} & =
\frac{\partial C}{\partial A}
\frac{\partial A}{\partial Z}
\frac{\partial Z}{\partial W} \\

\frac{\partial C}{\partial b} & =
\frac{\partial C}{\partial A}
\frac{\partial A}{\partial Z}
\frac{\partial Z}{\partial b}
\end{align}
$$


### Step 1: Derivative of Cost Function
First, we take the derivative of the cost function with respect to the output.

$$\frac{\partial C}{\partial A^{[L]}} = \dots$$

If we are using mean-squared error, then the derivative is simply

$$\frac{\partial C}{\partial A^{[L]}} = A^{[L]} - Y$$

If we are using the cross-entropy cost function, then the derivative is

$$
\frac{\partial C}{\partial A^{[L]}} =
- \bigg( \frac{Y}{A^{[L]}} - \frac{(1 - Y)}{1 - A^{[L]}} \bigg)
$$

We can write a class to represent our cost function. Put this in `cost_function.py`.

{% highlight python %}
class CostFunction():
    """The cost function of our network."""

    def __init__(self, name):
        """Initialize which function to use based on name.

        Arguments:
            name (str): The name of the funciton to use. Use "MSE" or "CE".
        """
        if name not in ("MSE", "CE"):
            raise ValueError("Name must be MSE or CE")
        self.name = name


    def compute_cost(self, Y, AL):
        """Returns the cost for the given target and output values.

        Arguments:
            Y: The target values for the given examples.
            AL: The output values from the network.
        """
        if name == "MSE":
            return compute_cost_MSE(Y, AL)
        if name == "CE":
            return compute_cost_CE(Y, AL)


    def compute_derivative(self, Y, AL):
        """Returns the derivative of the cost w.r.t. the ouputs.

        Arguments:
            Y: The target values for the given examples.
            AL: The output values from the network.
        """
        if name == "MSE":
            return compute_derivative_MSE(Y, AL)
        if name == "CE":
            return compute_derivative_CE(Y, AL)



# --------------------------------------------------
# Mean Squared Error
# --------------------------------------------------

def compute_cost_MSE(Y, AL):
    """Computes the cost using the MSE cost function."""
    m = Y.shape[1]
    cost = (1/(2*m)) * np.sum(np.square(Y - AL))
    return np.squeeze(cost)

def compute_derivative_MSE(Y, AL):
    """Computes the derivative of the cost w.r.t output using MSE."""
    return AL - Y


# --------------------------------------------------
# Cross Entropy
# --------------------------------------------------

def compute_cost_CE(Y, AL):
    """Computes the cost using the CE cost function."""
    m = Y.shape[1]
    cost = (-1/m) * np.sum(np.multiply(np.log(AL + eps), Y) + np.multiply(np.log((1-AL) + eps),(1-Y)))
    return np.squeeze(cost)

def compute_derivative_CE(Y, AL):
    """Computes the derivative of the cost w.r.t output using CE."""
    return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
{% endhighlight %}

Then, we can start working on a backpropagation method in `network.py`.

{% highlight python %}
def backpropagation(self, AL, Y, caches):
    """Perform backpropagation.

    Arguments:
        AL: The output of our final layer.
        Y: The target outputs of our network.
        caches: The caches output during forward propagation.

    Returns:
        gradients: A dictionary with the gradients
    """
    # Initialize some variables and the gradients dictionary
    gradients = {}
    L = self.num_layers - 1
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Step one: Compute the derivative of the cost function.
    dAL = self.cost_function.compute_derivative(AL, Y)
{% endhighlight %}


## Step 2: Derivatives with Respect to $$Z^{[l]}$$
As mentioned previously, in forward propagation we go from $$A^{[l-1]}$$, to $$Z^{[l]}$$, to $$A^{[l]}$$, but in backpropagation we go from $$A^{[l]}$$, to $$Z^{[l]}$$, to $$A^{[l-1]}$$. We just computed the derivative with respect to $$A^{[L]}$$ which we are calling `dAL`, so now we will compute the derivative with respect to $$Z^{[L]}$$.

Recall the formula

$$A^{[l]} = g(Z^{[l]})$$

Then the derivative is simply

$$\frac{dA^{[l]}}{dZ^{[l]}} = g'(Z^{[l]})$$

And from the chain rule mentioned previously

$$
\frac{\partial C}{\partial Z} =
\frac{\partial C}{\partial A}
\frac{\partial A}{\partial Z}
$$

This gives us our final formula for this step

$$
\frac{\partial C}{\partial Z} = dZ^{[l]} =
dA^{[l]} * g'(Z^{[l]})
$$


### Step 3: Derivatives with Respect to $$W^{[l]}$$
The moment we've been waiting for -- the derivative with respect to our weights!

Recall the formula for $$Z^{[l]}$$ is as follows

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

Then the derivative is as follows (it's a regular derivative with a transpose... [why?](https://math.stackexchange.com/questions/2044191/given-the-matrix-a-and-vector-x-what-is-the-partial-derivative-of-ax-with-respe))

$$\frac{dZ^{[l]}}{dW^{[l]}} = A^{[l-1]T}$$

Then referring back to our chain rule

$$
\frac{\partial C}{\partial W} =
\frac{\partial C}{\partial A}
\frac{\partial A}{\partial Z}
\frac{\partial Z}{\partial W}
$$

Since we've already computed the derivative with respect to $$Z^{[l]}$$

$$
\frac{\partial C}{\partial W} =
\frac{\partial C}{\partial Z}
\frac{\partial Z}{\partial W}
$$

Which gives us the final formula for this step

$$
\frac{\partial C}{\partial W} = dW^{[l]} =
\frac{1}{m} dZ^{[l]} A^{[l-1]T}
$$


### Step 4: Derivative with Respect to $$b^{[l]}$$
The other moment we've been waiting for -- the derivative with respect to our bias!

Recall the formula for $$Z^{[l]}$$ is as follows

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

Then the derivative is as follows (it's a regular derivative this time)

$$\frac{dZ^{[l]}}{db^{[l]}} = 1$$

Then referring back to our chain rule

$$
\frac{\partial C}{\partial b} =
\frac{\partial C}{\partial A}
\frac{\partial A}{\partial Z}
\frac{\partial Z}{\partial b}
$$

Since we've already computed the derivative with respect to $$Z^{[l]}$$

$$
\frac{\partial C}{\partial b} =
\frac{\partial C}{\partial Z}
\frac{\partial Z}{\partial b}
$$

Which gives us the final formula for this step

$$
\frac{\partial C}{\partial b} = db^{[l]} =
\frac{1}{m} dZ^{[l]}
$$


### Step 5: On to the Next Layer!
Now that we have found the derivative with respect to our weights and our bias, our network can start to learn. However, we don't want to stop at just one layer, we need the entire network to learn. To reiterate one last time, in forward propagation we go from $$A^{[l-1]}$$, to $$Z^{[l]}$$, to $$A^{[l]}$$, but in backpropagation we go from $$A^{[l]}$$, to $$Z^{[l]}$$, to $$A^{[l-1]}$$. This means we now need to find the derivative with respect to $$A^{[l-1]}$$. Then, we repeat steps 2 through 5 until we reach the input layer.

Recall the formula for $$Z^{[l]}$$ is as follows

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

Then the derivative is as follows (it's _almost_ a regular derivative)

$$\frac{dZ^{[l]}}{dA^{[l-1]}} = W^{[l]T}$$

Then referring back to our chain rule

$$
\frac{\partial C}{\partial A^{[l-1]}} =
\frac{\partial C}{\partial A}
\frac{\partial A}{\partial Z}
\frac{\partial Z}{\partial A^{[l-1]}}
$$

Since we've already computed the derivative with respect to $$Z^{[l]}$$

$$
\frac{\partial C}{\partial A^{[l-1]}} =
\frac{\partial C}{\partial Z}
\frac{\partial Z}{\partial A^{[l-1]}}
$$

Which gives us the final formula for backpropagation

$$
\frac{\partial C}{\partial A^{[l-1]}} = dA^{[l-1]} =
W^{[l]T} dZ^{[l]}
$$



<hr>
## Finishing our Backpropagation Implementation
Here is the completed implementation of backpropagation for `network.py`

{% highlight python %}
def backpropagation(self, AL, Y, caches):
    """Perform backpropagation.

    Arguments:
        AL: The output of our final layer.
        Y: The target outputs of our network.
        caches: The caches output during forward propagation.

    Returns:
        gradients: A dictionary with the gradients
    """
    # Initialize some variables and the gradients dictionary
    gradients = {}
    L = self.num_layers - 1
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Step one: Compute the derivative of the cost function.
    dAL = self.cost_function.compute_derivative(AL, Y)

    # Perform steps two through five for the output layer
    cache = caches[L-1]
    activation = self.activations[L-1]
    gradients["dA" + str(L-1)], gradients["dW" + str(L)], gradients["db" + str(L)] = self.backward_step(dAL, cache, activation)

    # All other layers
    for l in reversed(range(L-1)):
        cache = caches[l]
        activation = self.activations[l-1]
        dA_prev, dW, db = self.backward(gradients["dA" + str(l + 1)], current_cache, activation)
        gradients["dA" + str(l)] = dA_prev
        gradients["dW" + str(l + 1)] = dW
        gradients["db" + str(l + 1)] = db

    return gradients


def backward_step(self, dA, cache, activation):
    """Perform backpropagation for a single layer.

    Arguments:
        dA: Gradient for the current layer l
        cache: A tuple (W, A_prev, b, Z, A)
        activation: The activation function for the current layer.

    Returns:
        dA_prev: Gradient with respect to the previous activations
        dW: The gradient of the cost with respect to W
        db: Gradient of the cost with respect to b
    """
    (W, A_prev, b, Z, A) = cache
    m = A_prev.shape[1]
    dZ = None

    # Compute dZ
    dZ = dA * activation.derivative(Z)

    # Compute dA_prev, dW, and db
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db
{% endhighlight %}



<hr>
## After Reading This Post
After reading this post, you should be able to answer the following questions.
1. What is a cost function?
2. What properties should a cost function have?
3. What are the backpropagation equations?
4. What outputs can we cache during forward propagation for use in backpropagation?


### What is a cost function?
A cost function measures how "right" or "wrong" our network was on the training set. A larger value means our network was really wrong, and a smaller value means our network. Our goal is to minimize the cost function -- and that's what we mean when we say our network is "learning."

### What properties should a cost function have?
A cost function should have these two properties
1. It should be an average of the errors on each individual example
2. It should be a function of the weights and biases in our network

### What are the backpropagation equations?
The first equation is derivative of the cost function with respect to our outputs. This will change depending on what cost function we use. For cross-entropy cost we will use the following formula

$$\frac{\partial C}{\partial A^{[L]}} =
- \bigg( \frac{Y}{A^{[L]}} - \frac{(1 - Y)}{1 - A^{[L]}} \bigg)
$$

The second equation is the derivative with respect to $$Z^{[l]}$$

$$
\frac{\partial C}{\partial Z} = dZ^{[l]} =
dA^{[l]} * g'(Z^{[l]})
$$

The third equation is the derivative with respect to $$W^{[l]}$$

$$
\frac{\partial C}{\partial W} = dW^{[l]} =
\frac{1}{m} dZ^{[l]} A^{[l-1]T}
$$

The fourth equation is the derivative with respect to $$b^{[l]}$$

$$
\frac{\partial C}{\partial b} = db^{[l]} =
\frac{1}{m} dZ^{[l]}
$$

The fifth equation is the derivative with respect to $$A^{[l-1]}$$

$$
\frac{\partial C}{\partial A^{[l-1]}} = dA^{[l-1]} =
W^{[l]T} dZ^{[l]}
$$

### What outputs can we cache during forward propagation?
Looking at the above equations, these are the values that we computed during forward propagation and reused in backpropagation. During training, we should cache these values for use in the backpropagation phase.
+ $$Z^{[l]}$$: The pre-activation output of each layer
+ $$A^{[l]}$$: The activations of each layer
+ $$W^{[l]}$$: The weights of our network


<hr>
## Coming Up
Today, we implemented the backpropagation algorithm. There is one problem though -- we never used it to updates the weights and biases! In the next post, we are going to finish implementing our neural network by implementing the update rule and the training algorithm. After that we are going to use our neural network to recognize handwritten digits. 

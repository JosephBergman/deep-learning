---
layout: post
title: "Forward Propagation"
published: 2018-10-14
updated: 2018-10-14
preview: "Learn to efficiently compute outputs in a deep neural network"
---

In the previous post, we looked at what neural networks are and how they compute their output. In this post, we will devise a more concise notation for computing a neural network's output, and we will start to write some code for a neural network in Python. This post will require some familiarity with both [Python](http://www.diveintopython3.net/) and [Linear Algebra](https://www.khanacademy.org/math/linear-algebra).



<hr>
## After Reading This Post

![A 3-Layer Neural Network]({{relative_url}}/assets/post02/nn-1.svg){:width="300"}
_Figure 1_

After reading this post you should be able to answer the following questions:
1. What are the dimensions of $$W^{[1]}$$, $$W^{[2]}$$, and $$W^{[3]}$$ for this network?
2. What are the dimensions of $$b^{[1]}$$, $$b^{[2]}$$, and $$b^{[3]}$$ for this network?
3. How are $$z^{[i]}$$ and $$a^{[i]}$$ different?
4. How are $$z^{[i]}$$ and $$a^{[i]}$$ different from $$Z^{[i]}$$ and $$A^{[i]}$$?

The answers are provided at the end!



<hr>
## The Weight Matrix $$W^{[i]}$$
In the previous post, when we computed the output of our neural network, we took the inputs and fed them into each neuron individually. We were able to represent our inputs and weights as vectors, so we could simplify the computation for each neuron to the following dot product.

$$z = w^T x \quad \text{($+b$, which we are ignoring for now)}$$

Doing this for every neuron individually is rather tedious. The value of $$x$$ never changes -- we just multiply it by different weights repeatedly. It turns out we can compute this dot product for every neuron in a layer at once by putting the weight vectors into a matrix $$W$$. Then, all we have to do is compute the following

$$z = W x$$


### A Practice Example
![Our example network]({{relative_url}}/assets/post02/nn-example.svg){:height="175"}

Previously, we were computing three separate dot products for the first layer.

$$
\begin{align}
z_{h1} & = w^T x \\
& = 0.25\cdot x_1 + -0.25\cdot x_2
\end{align}
$$

$$
\begin{align}
z_{h2} & = w^T x \\
  & = -0.5\cdot x_1 + 0.5\cdot x_2
\end{align}
$$

$$
\begin{align}
z_{h2} & = w^T x \\
  & = -1\cdot x_1 + 1\cdot x_2
\end{align}
$$

Instead, we can just perform a single matrix-vector multiplication

$$
\begin{align}
z &= W x \\
& =
\begin{bmatrix}
0.25 & -0.25 \\
-0.5 & 0.5 \\
-1.0 & 1.0 \\
\end{bmatrix}

\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix}

\end{align}
$$

Now, instead of being a single value, $$z$$ will also be a vector.

$$z =  
\begin{bmatrix}
-0.25 \\
0.5 \\
1.0 \\
\end{bmatrix}
$$

We will have a weight matrix for each layer of our neural network, so we will label them with a bracketed superscript. Therefore, the example shown above can be written concisely as

$$z = W^{[1]} x$$

In practice, we will give all of our weight matrices random weights. Therefore, the only thing you need to remember about the weight-matrix is its dimensions -- you don't have to worry about what weight goes where. Below is a summary of what you need to remember.

### Weight Matrix Dimensions
The weight matrix $$W^{[l]}$$ has dimensions $$n^{[l]} \times n^{[l-1]}$$ where $$n^{[l]}$$ is the number of units in layer $$l$$. This is easy to remember for two reasons: (1) It has to have $$n_l$$ rows so that there is one output per neuron, and (2) It has to have $$n_{l-1}$$ columns or we couldn't multiply it by its input -- the dimensions would not match!



<hr>
## The Bias Vector $$b^{[i]}$$
As you may recall, each neuron also has a bias associated with it, which is just some scalar value. In order to "linear algebra-ify" our neural network computations, we will put all the biases into a vector. So originally we computed the following for each neuron

$$z = w^T x + b$$

And now we will compute the following

$$z = W^{[l]}x + b^{[l]}$$

This isn't too different from what we did before. The only difference is that $$b^{[l]}$$ is an $$n^{[l]} \times 1$$ column vector containing the biases for each neuron in layer $$l$$.


### A Practice Example
![Our example network with biases]({{relative_url}}/assets/post02/nn-example-bias.svg){:height="175"}
_Example with Biases_

Here is a quick example with biases added. The dimensions of $$Wx$$ and $$b$$ have to match.

$$
\begin{align}
z &= W^{[1]} x + b^{[1]} \\

&=
\begin{bmatrix}
0.25 & -0.25 \\
-0.5 & 0.5 \\
-1.0 & 1.0 \\
\end{bmatrix}

\begin{bmatrix}
x_1 \\
x_2 \\
\end{bmatrix}

+
\begin{bmatrix}
1 \\
2 \\
3 \\
\end{bmatrix} \\

&=
\begin{bmatrix}
-0.25 \\
0.5 \\
1.0 \\
\end{bmatrix}

+
\begin{bmatrix}
1 \\
2 \\
3 \\
\end{bmatrix} \\

&=
\begin{bmatrix}
0.75 \\
2.5 \\
4.0 \\
\end{bmatrix}

\end{align}
$$



<hr>
## The Layer Outputs $$z^{[i]}$$ and $$a^{[i]}$$
So far, we have been looking at computing the outputs of just the first layer. Furthermore, we have only been looking at how to calculate $$z$$. We have to apply some activation function (ReLU, Sigmoid, etc.) to the output of every neuron to get the activation of the layer, $$a$$.

In the first layer, we will begin by computing the following

$$z^{[1]} = W^{[1]} x + b^{[1]}$$

Then, we apply some activation function $$g(\cdot)$$ to get

$$a^{[1]} = g(z)$$

This activation vectors, $$a^{[1]}$$, will then be the input to the next layer, giving us the following formula for the remaining layers in our network.

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} \qquad \text{for all $l > 2$}$$

To simplify this formula, we are going to refer to the input $$x$$ as $$a^{[0]}$$ from now on. This gives us just two formulas to remember when computing the output of our neural network.

$$
\begin{align}
z^{[l]} & = W^{[l]} a^{[l-1]} + b^{[l]} \\
a^{[l]} & = g(z)
\end{align}
$$


### A Practice Example
![Our example network with biases]({{relative_url}}/assets/post02/nn-example-bias.svg){:height="175"}
_A Full Example_

Using our new notation, let's compute the output of this network.

First we compute the output of layer one.

$$
\begin{align}
z^{[1]} & = W^{[1]} a^{[0]} + b^{[1]} \\

& =
\begin{bmatrix}
0.25 & -0.25 \\
-0.5 & 0.5 \\
-1.0 & 1.0 \\
\end{bmatrix}

\begin{bmatrix}
1 \\
2 \\
\end{bmatrix}

+
\begin{bmatrix}
1 \\
2 \\
3 \\
\end{bmatrix} \\

& =

\begin{bmatrix}
-0.25 \\
0.5 \\
1.0 \\
\end{bmatrix}

+
\begin{bmatrix}
1 \\
2 \\
3 \\
\end{bmatrix} \\

&=
\begin{bmatrix}
0.75 \\
2.5 \\
4.0 \\
\end{bmatrix} \\

a^{[1]} &= ReLU(z^{[1]}) \\

&=
\begin{bmatrix}
0.75 \\
2.5 \\
4.0 \\
\end{bmatrix}
\end{align}
$$

Next, we compute the output of the output layer.

$$
\begin{align}
z^{[2]} & = W^{[2]} a^{[1]} + b^{[2]} \\

& =
\begin{bmatrix}
-1 & 0 & 1 \\
\end{bmatrix}

\begin{bmatrix}
0.75 \\
2.5 \\
4.0 \\
\end{bmatrix}

+
\begin{bmatrix}
-2 \\
\end{bmatrix} \\

& =
\begin{bmatrix}
1.25 \\
\end{bmatrix} \\

a^{[2]} &= \sigma(z^{[2]}) \\

&=
\sigma([1.25]) \\

& \approx
0.7772999
\end{align}
$$

So the output of our network for the given weights, biases, and inputs is approximately $$0.7772999$$.



<hr>
## Working with Multiple Examples
This is the last thing we need to discuss before we can begin implementing our neural network. In the previous example, our neural network was given just one set of inputs and was asked to give the corresponding output. In practice, however, we will want to show our neural network thousands (sometimes millions) of examples for it to learn from. Unlike humans, neural networks need to see lots of examples to learn. Therefore, we need an efficient way to feed thousands of examples into our network.


### The Importance of Vectorization
If you have a programming background, you might be thinking, "Let's  use a for-loop and feed all of the examples through one at a time." This would work, but it would be very slow. When working with neural networks, we want to avoid for-loops as much as possible and write vectorized code. We usually avoid for-loops by using built-in functions from the `numpy` library. As an example of how much faster vectorized code is, here is an example which computes the dot product of two vectors.

{% highlight python %}
import numpy as np
import time

# Generate two random vectors
n = 10000000
x = np.random.randn(n)
y = np.random.randn(n)

# Compute their dot-product with a for-loop
tic = time.time()
z1 = 0
for i in range(n):
    z1 += x[i] * y[i]
toc = time.time()
time1 = toc - tic
print("For Loop")
print("--------")
print("Out: " + str(z1))
print(str(time1*1000) + "ms\n\n")

# Compute their dot-product with numpy
tic = time.time()
z2 = np.dot(x,y)
toc = time.time()
time2 = toc - tic
print("Vectorized")
print("----------")
print("Out: " + str(z2))
print(str(time2*1000) + "ms\n\n")

# How different are the results?
print("Difference")
print("----------")
print("For Loop is " + str (time1/time2) + " times slower.")
{% endhighlight %}

And here is the output of that program
{% highlight python %}
For Loop
--------
Out: -569.032297628
4745.116949081421ms

Vectorized
----------
Out: -569.032297628
39.34192657470703ms

Difference
----------
For Loop is 120.6122160812547 times slower.
{% endhighlight %}

To put that in perspective, 120 times slower could be the difference between a neural network that takes one minute to train and a neural network that takes two hours to train. To really put that in perspective, could you imagine paying to train a model in the cloud for two hours when you could have trained it in just one minute? For more explanation on _why_ vectorized code is faster, check out [this StackOverflow post](https://stackoverflow.com/questions/35091979/why-is-vectorization-faster-in-general-than-loops).


### Feeding in Multiple Examples
Rather than using a for-loop to feed in multiple examples, we are going to make one final modification to our formulas. Currently, we are using the following equations for computing the output of a network. In these formulas, everything is a column vector except for the weight matrix.

$$
\begin{align}
z^{[l]} & = W^{[l]} a^{[l-1]} + b^{[l]} \\
a^{[l]} & = g(z)
\end{align}
$$

Now, we are going to replace our input vector $$a^{[0]}$$ with an input matrix $$A^{[0]}$$. The matrix $$A^{[0]}$$ will have dimensions $$n_0 \times m$$ -- in other words, we will take all of our examples, and put them in a single matrix arranged by columns. As a result, the dimensions of our other variables will change.
+ $$W^{[l]}$$ is still an $$(n^{[l]} \times n^{[l-1]})$$ matrix.
+ $$A^{[l-1]}$$ is an $$(n^{[l-1]} \times m)$$ matrix.
+ $$Z^{[l]}$$ therefore, is an $$(n^{[l]} \times m)$$ matrix.
+ $$A^{[l]}$$ will then be an $$(n^{[l]} \times m)$$ matrix as well.

However, $$b^{[l]}$$ actually does not change. It is still a column vector with dimensions $$(n^{[l]} \times 1)$$. Adding $$b^{[l]}$$, an $$(n^{[l]} \times 1)$$ vector, to $$W^{[l]} A^{[l-1]}$$, an $$(n^{[l]} \times m)$$ matrix, is technically not defined. What we're actually going to do is add the vector to each column of the matrix. We could duplicate the bias vector $$m$$ times to create a matrix with the correct dimensions, but this is unnecessary. Since the number of rows in $$b$$ matches the number of rows in $$W^{[l]} A^{[l-1]}$$ Python will use [broadcasting](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting) to automatically add the vector to each column.


### Our Final Formulas
Based on the previous changes, we now have the following formulas.

$$
\begin{align}
Z^{[l]} & = W^{[l]} A^{[l-1]} + b^{[l]} \\
A^{[l]} & = g(Z^{[l]})
\end{align}
$$

The dimensions are as follows
+ $$W^{[l]}$$ is an $$(n^{[l]} \times n^{[l-1]})$$ matrix.
+ $$A^{[l-1]}$$ is an $$(n^{[l-1]} \times m)$$ matrix.
+ $$Z^{[l]}$$ is an $$(n^{[l]} \times m)$$ matrix.
+ $$A^{[l]}$$ is an $$(n^{[l]} \times m)$$ matrix.
+ $$b^{[l]}$$ is an $$(n^{[l]} \times 1)$$ column vector



<hr>
## Implementing Forward Propagation
Finally, we have a concise mathematical notation for how to compute the output of our neural network. The algorithm described above is known as _forward propagation_ and it is the first step to training our neural network. In this section, we will start to implement a neural network from scratch using Python. In the following posts, we will implement _back propagation_, complete our neural network, and learn how to recognize handwritten digits using our network.

You can follow along with this post (and the following posts) using [this Jupyter Notebook](). If you need to refer to the final code at any point, you can [find it here]().


### Initializing Our Neural Network
Let's begin by defining a `Network` class in `network.py`

{% highlight python %}
import numpy as np

class Network:
    """A vectorized, L-layer neural network."""

    def __init__(self):
        pass
{% endhighlight %}

We want to be able to easily modify the architecture of our neural network, so let's allow the user to pass in the desired layer sizes as a list. We are going to store our weight matrices and bias vectors in a dictionary called `parameters`. Replace the `__init__` function with the following.

{% highlight python %}
def __init__(self, layer_sizes):
    """Initializes a neural network with the given dimensions.

    Args:
        layer_sizes (list): Number of units in each layer
    """
    self.layer_sizes = layer_sizes
    self.num_layers = len(layer_sizes)
    self.parameters = {}
    self.reset()
{% endhighlight %}

At the end of our `__init__()` function we call a method `reset()`. This method will set our weights and biases to their initial values. In order for our neural network to learn well, we want to initialize the weights to random values. The biases' initial values don't matter as much, so we'll just set them to zero. Recall the expected dimension of each variable, and see how it is reflected in the code.

{% highlight python %}
def reset(self):
    """Randomly initializes all the weights and sets the biases to 0."""
    layer_sizes = self.layer_sizes
    for l in range(1, self.num_layers):
        self.parameters['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.01
        self.parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))
{% endhighlight %}


### Forward Propagation for a Single Layer
Before performing forward propagation for our entire network, let's first write the code for a single layer. Recall the formulas for a single layer.

$$
\begin{align}
Z^{[l]} & = W^{[l]} A^{[l-1]} + b^{[l]} \\
A^{[l]} & = g(Z^{[l]})
\end{align}
$$

These formulas can be easily implemented in code using `numpy`. We will assume that the activation function is passed in to the method and takes a single argument `Z`. You can ignore the cache for now -- it will simplify our training algorithm and will be explained two posts from now.

{% highlight python %}
def forward(self, A_prev, W, b, activation):
    """Computes a single layer's output.

    Args:
        A_prev: The inputs to the layer
        W: The weights of the layer
        b: The biases of the layer
        activation: The activation function to use

    Returns:
        A: The activation of the layer
        cache: A tuple (W, A_prev, b, Z, A)
    """
    Z = np.dot(W,A_prev) + b
    A = activation(Z)
    cache = (W, A_prev, b, Z, A)
    return A, cache
{% endhighlight %}


### Implementing Forward Propagation
Now we can implement forward propagation for our entire network. I know earlier I said to avoid for-loops, but this is one place where we can't really help it. For now, we are going to assume that all of our hidden units are using ReLU activation and our output units are using sigmoid activation. Because of this, we will handle our hidden layers and our output layer separately. Refer to the following code.

{% highlight python %}
def forward_propagation(self, X, parameters):
    """Performs forward propagation for the given examples.

    Args:
        X: The inputs to the NN
        parameters: The weights and biases of the NN

    Returns:
        AL: The output of the neural network
        caches: A list of cached computations
    """
    # Set A^[0] to be X for cleaner notation
    A = X
    L = self.num_layers - 1
    parameters = self.parameters
    self.caches = []

    # Hidden layer computations
    for l in range(1, L):
        A_prev = A
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        A, cache = self.forward(A_prev, Wl, bl, relu)
        self.caches.append(cache)

    # Output layer computations
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    AL, cache = self.forward(A, WL, bL, sigmoid)
    self.caches.append(cache)

    return AL, caches
{% endhighlight %}


### Activation Functions
Finally, our `forward_propagation()` function uses two currently undefined functions: `relu` and `sigmoid`. We can define these outside of our class or in a separate file. It is very important that we use `numpy` to implement these functions so that they can take vectors as inputs. Here are their implementations in Python.

{% highlight python %}
def sigmoid(Z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-Z))

def relu(Z):
    """The ReLU function."""
    return np.maximum(Z, 0)
{% endhighlight %}


### Conclusion
And, that's it. Right now we have a neural network with a fully implemented forward-propagation algorithm. In the next post, we will see how to implement _back-propagation_ which allows us to train our neural network. After that, we will combine these functions into a single training function and we will make some minor improvements to our network. Finally, we will put our neural network to the test recognizing handwritten digits. Before moving on, test your understanding with the "After Reading This Post" questions.


<hr>
## After Reading This Post

![A 3-Layer Neural Network]({{relative_url}}/assets/post02/nn-1.svg){:width="300"}
_Figure 1_

After reading this post you should be able to answer the following questions:
1. What are the dimensions of $$W^{[1]}$$, $$W^{[2]}$$, and $$W^{[3]}$$ for this network?
2. What are the dimensions of $$b^{[1]}$$, $$b^{[2]}$$, and $$b^{[3]}$$ for this network?
3. How are $$z^{[i]}$$ and $$a^{[i]}$$ different?
4. How are $$z^{[i]}$$ and $$a^{[i]}$$ different from $$Z^{[i]}$$ and $$A^{[i]}$$?

Here are the answers
1. $$(5 \times 2)$$, $$(5 \times 5)$$, and $$(1 \times 5)$$ respectively
2. $$(5 \times 1)$$, $$(5 \times 1)$$, and $$(1 \times 1)$$ respectively
3. $$z^{[i]}$$ is the result of computing $$W^{[l]} a^{[l-1]} + b^{[i]}$$, but has not applied an activation function. $$a^{[i]}$$ is simply the result of applying an activatio function to $$z^{[i]}$$.
4. $$z^{[i]}$$ and $$a^{[i]}$$ are simply column vectors. We get column vectors when we pass a single example through the network at a time. In practice, we will pass an example matrix containing $$m$$ examples into the network. As a result, we will have $$Z^{[i]}$$ and $$A^{[i]}$$ which are matrices containing $$m$$ columns.

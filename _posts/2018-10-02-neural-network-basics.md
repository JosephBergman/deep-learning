---
layout: post
title: "Neural Network Basics"
published: 2018-10-02
updated: 2018-10-02
preview: "What do neural networks do and how do they do it?"
---

Throughout the next few posts we will answer the following three questions:

1. What is a neural network?
2. What does a neural network compute?
3. How can we teach a neural network to "learn" from examples?

This post will focus on the first question.



<hr>
## What is a Neural Network?
An Artificial Neural Network (ANN), or simply "neural network", is a biologically-inspired programming paradigm which enables a program to learn from data without being explicitly programmed. For example, we can show a neural network thousands of cat/not-cat photos and the neural network can learn on its own to predict whether or not a given photo contains a cat.

Similar to the human brain, a neural network is composed of many smaller and simpler units known as "(artificial) neurons". Individually, each neuron performs a relatively simple computation, but by feeding the output of one neuron into other neurons we can compute progressively more complex functions. Neural networks currently provide the best solutions to a number of problems in image recognition, speech recognition, and natural language processing.

![Example of a Neural Network](/deep-learning/assets/post01/nn-example.svg){:width="200px"}
_An example of a 5-layer neural network_

More precisely, a neural network is a map from some input space to a range of output values. The input space and output space will vary depending on the problem. For instance, the input could be a photo of a handwritten digit and the output would be some digit between 0 and 9. In a more complex case, the input could be some photo and the output could be a caption for that photo. What makes neural networks so powerful is that they can learn the mapping between the input and output just by looking at examples.

Before we can understand how a neural network learns, we need to understand what a neural network does to map a given input to an output. And before we can understand how a neural network maps an input to an output, we need to understand what a single neuron does. We will begin by looking at the simplest type of artificial neuron -- the perceptron.



<hr>
## The Perceptron
A perceptron takes an arbitrary number of inputs $$x_1, x_2, \dots$$ and generates a single binary output $$a \in \{0, 1\}$$. Each input $$x_i$$ has a corresponding weight $$w_i$$ which expresses how important that input is to the output. The perceptron also has a bias $$b$$ which is a single real number and expresses how willing the perceptron is to "fire" (produce an output of 1).

The perceptron computes its output in two steps. First, we multiply all the inputs by their corresponding weights and add the bias. This can be simplified by treating $$x_i$$ and $$w_i$$ as column vectors and taking their dot product:
$$z = w^T x + b$$

Then, we compute the output according to the following function:

$$g(z) =
\begin{cases}
1, & \text{if $z > 0$} \\
0, & \text{if $z \leq 0$}
\end{cases}
$$

The function $$g(z)$$ is an example of an "activation function", and will be the only thing that changes in our other neurons.


### Example of a Perceptron
![Example of a Perceptron](/deep-learning/assets/post01/perceptron-example.png)
_([source](http://neuralnetworksanddeeplearning.com/chap1.html))_

The perceptron above takes two inputs $$x_1$$ and $$x_2$$ which have corresponding weights $$w_1 = -2$$ and $$w_2 = -2$$ as shown. The perceptron has a bias of $$b = 3$$. The value of the output depends on the specific values of the inputs. For example, suppose $$x_1 = 1$$ and $$x_2 = 0$$. Then

$$x =
\begin{bmatrix}
1 \\
0 \\
\end{bmatrix}

\qquad

w =
\begin{bmatrix}
-2 \\
-2 \\
\end{bmatrix}

\qquad
b = 3
$$

It follows that

$$
\begin{align}
z & = w^T x + b \\
  & = (-2\cdot 1 + -2\cdot 0) + 3 \\
  & = (-2) + 3 \\
  & = 1
\end{align}
$$

Since this is positive our activation function gives us the following

$$
\begin{align}
\text{output} & = g(z) \\
  & = g(1) \\
  & = 1 \quad \text{since $z > 0$}
\end{align}
$$


### Problems with the Perceptron
Ultimately we want to devise algorithms which allow our neural network to learn the weights $$w$$ and biases $$b$$ on its own. To be able to learn, we want a small change to the weights or bias to cause a small change in the corresponding output. If we can achieve this, we can gradually update our weights and biases until our outputs are very accurate. Unfortunately, perceptrons don't have this feature.

![Perceptron Activation Function](/deep-learning/assets/post01/perceptron-activation.png){:height="200"}
_Perceptron Activation Function ([Source](https://www.codeproject.com/Articles/1216170/Common-Neural-Network-Activation-Functions))_

Looking at the perceptron activation function, we can see that a small change in the weights or biases can have one of two effects:
1. The output won't change at all
2. The output will change completely

This is because the perceptron activation function is (1) not continuous and (2) has a derivative of zero when it is differentiable. We want activation functions that are both continuous and differentiable. In the next section, we will look at four different activation functions that we can use with neurons in our neural network.



<hr>
## Activation Functions
In this section we will look at four different activation functions, $$g(\cdot)$$:
1. The sigmoid function: $$\sigma(z)$$
2. The hyperbolic tangent function: $$\tanh(z)$$
3. The ReLU function: $$\text{ReLU}(z)$$
4. The Leaky ReLU function: $$\text{LeakyReLU}(z)$$


### The Sigmoid Function
![Sigmoid Activation Function](/deep-learning/assets/post01/sigmoid-activation.png){:height="200"}
_Sigmoid Activation Function ([Source](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e))_

The sigmoid function has the following formula

$$\sigma(z) = \frac{1}{1 + e^(-z)}$$

The sigmoid function has the following derivate ([derivation](http://kawahara.ca/how-to-compute-the-derivative-of-a-sigmoid-function-fully-worked-example/))

$$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

The sigmoid function is both continuous and differentiable as desired. Another useful property of the sigmoid function is that the output of the sigmoid function is always between 0 and 1, so we can interpret the output of the function as a probability.


### The Hyperbolic Tangent Function
![Hyperbolic Tangent Activation Function](/deep-learning/assets/post01/tanh-activation.jpg){:height="200"}
_Hyperbolic Tangent Activation Function ([Source](http://www.20sim.com/webhelp/language_reference_functions_tanh.php))_

The hyperbolic tangent function has the following formula

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

The hyperbolic tangent function has the following derivate

$$\tanh'(z) = 1 - (tanh^2(z))^2$$

The hyperbolic tangent function is continuous and differentiable as desired. Another useful property of the hyperbolic tangent function is that the output is between -1 and 1 and centered around 0 (see graph above). The result is that the hyperbolic function has a way of "centering" our data about 0 which tends to cause our neural networks to learn faster.

In course one of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) Professor Ng says that hyperbolic tangent is almost always better than the sigmoid function, except for possibly in the output layer where we may want to use the sigmoid function to represent a probability.


### The Rectified Linear Unit (ReLU)
![ReLU Activation Function](/deep-learning/assets/post01/relu-activation.png){:height="200"}
_ReLU Activation Function ([Source](https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/))_

The ReLU has a very simple formula

$$\text{ReLU}(z) = \max(0, z)$$

The derivative of the ReLU is as follows

$$\text{ReLU}'(z) =
\begin{cases}
1, & \text{if $z > 0$} \\
0, & \text{if $z < 0$} \\
\text{DNE}, & \text{if z = 0, in practice use 0 or 1}
\end{cases}
$$

The ReLU function is continuous and differentiable everywhere except $$z = 0$$. In practice, this does not matter and you can set the derivative at $$z = 0$$ to be either 1 or 0. The ReLU function is currently the most used activation function.

If you look at the sigmoid and hyperbolic tangent functions, you will notice that for very large or very small values of $$z$$ the derivative is nearly 0. This will slow down learning, so the ReLU is preferred. Of course, the derivative of the ReLU is 0 for all $$z < 0$$ which leads to our final activation function.


### The Leaky ReLU
![Leaky ReLU Activation Function](/deep-learning/assets/post01/leaky-relu-activation.png){:height="200"}
_Leaky ReLU Activation Function ([Source](https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/))_

The Leaky ReLU also has a simple formula

$$\text{LeakyReLU}(z) = \max(0.01z, z)$$

The derivative of the Leaky ReLU is as follows

$$\text{LeakyReLU}'(z) =
\begin{cases}
1, & \text{if $z > 0$} \\
0.01, & \text{if $z < 0$} \\
\text{DNE}, & \text{if z = 0, in practice use 0.01 or 1}
\end{cases}
$$

The Leaky ReLU function is extremely similar to the ReLU function, but has a non-zero derivative for $$z < 0$$. The Leaky ReLU actually works quite well in practice, but it is rarely used compared to the standard ReLU function.



<hr>
## From Neurons to Neural Networks
Now that we have established what neurons are, how they compute their output, and what activation functions they can use, we can start to discuss neural networks. In this section, we will look at some neural network terminology, and we will compute the output of a neural network by hand. In the next post, we will establish some more mathematical notation to simplify this process.

![Neural Network Example](/deep-learning/assets/post01/nn-terminology-example.svg){:height="250"}


### Terminology
Above is a diagram for a simple 2-layer neural network. It may look like there are three layers, but the layer all the way on the left is the "Input Layer" and we don't count it. Thus the neural network shown above has 2-layers -- one hidden layer and one output layer. Since our end-user only cares about the input and the output, we call all the layers in between "hidden layers". An L-layer neural network will have L-1 hidden layers.

The nodes in the input layer, labeled $$x_1$$ and $$x_2$$ are not neurons, they are just real valued inputs. The hidden layer, however, contains three neurons and the output layer contains a single neuron.

Each neuron in the hidden layer takes two inputs, $$x_1$$ and $$x_2$$ and produces a single output. The neuron in the output layer takes three inputs, lets call them $$h_1$$, $$h_2$$, and $$h_3$$, and produces a single output $$y$$. Every edge in the neural network will have a weight associated with it, so this network has 9 weights in total.

To see how a neural network computes its output, let's give these variables some values and work through an example by hand. As mentioned, we will simplify the math and write some code to do this for us in the next post.


### Assigning Values to our Network
Let's suppose the inputs have the following values

$$x_1 = 1 \qquad x_2 = 2$$

Then we can create a column vector containing our inputs

$$x =
\begin{bmatrix}
1 \\
2 \\
\end{bmatrix}
$$

Let's give our hidden neurons some weights. Since each neuron takes two inputs, each neuron will need to have two weights. Suppose that $$h_i$$ has weights $$w_i$$ as follows

$$w_1 =
\begin{bmatrix}
0.25 \\
-.25 \\
\end{bmatrix}

\qquad

w_2 =
\begin{bmatrix}
-0.5 \\
0.5 \\
\end{bmatrix}

\qquad

w_3 =
\begin{bmatrix}
-1 \\
1 \\
\end{bmatrix}
$$

Our output neuron has three inputs, so it is going to need three weights. Let's suppose that our output neuron $$y$$ has the following weights

$$
w_y =
\begin{bmatrix}
-1 \\
0 \\
1 \\
\end{bmatrix}
$$

Finally, let's assume that the bias of each neuron is 0, and let's assume that the hidden neurons use ReLU activation and that the output neuron uses sigmoid activation.


### Computing the Output
With all of the values given above we have the following network

![Labeled Neural Network Example](/deep-learning/assets/post01/nn-example-labeled.svg){:height="250"}

In order to compute our final output, we first have to compute the output of each hidden neuron. We will essentially pass our input values "forward" through the network until we reach the output layer. The computations are below.

For our first hidden neuron

$$
\begin{align}
z_{h1} & = w^T x + b \\
  & = (0.25\cdot 1 + -0.25\cdot 2) + 0 \\
  & = -0.25 \\
a_{h1} & = \text{ReLU}(-0.25) \\
       & = 0
\end{align}
$$

For our second hidden neuron

$$
\begin{align}
z_{h2} & = w^T x + b \\
  & = (-0.5\cdot 1 + 0.5\cdot 2) + 0 \\
  & = 0.5 \\
a_{h1} & = \text{ReLU}(0.5) \\
       & = 0.5
\end{align}
$$

For our third hidden neuron

$$
\begin{align}
z_{h2} & = w^T x + b \\
  & = (-1\cdot 1 + 1\cdot 2) + 0 \\
  & = 1.0 \\
a_{h1} & = \text{ReLU}(1.0) \\
       & = 1.0
\end{align}
$$

We can combine the hidden layer outputs into an input vector.

$$
h =
\begin{bmatrix}
0 \\
0.5 \\
1.0 \\
\end{bmatrix}
$$

Our output layer is a single neuron with sigmoid activation, so we get our final output

$$
\begin{align}
z_{y} & = w^T h + b \\
  & = (-1\cdot 0 + 0\cdot 0.5 + 1\cdot 1) + 0 \\
  & = 1.0 \\
a_{y} & = \sigma(1.0) \\
       & \approx 0.73
\end{align}
$$

You can check the sigmoid function computation [here](https://www.wolframalpha.com/input/?i=sigmoid(1)).



<hr>
## Review
Right now, it may not clear how a neural network learns or why they're useful. For now, focus on the following takeaways.


### What is a neural network?
A neural network is a biologically-inspired programming paradigm that learns from examples without being explicitly programmed.

### What is a neural network made of?
A neural network is composed of many smaller units called neurons. Neurons weight their inputs, add a bias, and apply an activation function to create a single output. The output of one neuron can be fed into many other neurons allowing the neural network to learn complex mappings.

### What does a neural network do?
A neural network learns to map inputs to their desired outputs. For instance, given a photo output whether or not there is a cat in it. Another example could be, given some audio recording output a transcript in English.

### How does a neural network map an input to an output?
As we saw in the final example, a neural network passes its inputs through each layer of the network until they reach the output layer. The final output of a neural network will be determined by the weights, biases, and activations used within the network. In the next post, we will look at how we can efficiently map an input to its corresponding output.

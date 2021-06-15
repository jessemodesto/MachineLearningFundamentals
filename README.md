# Fundamentals of Machine Learning

## 1 - Introduction

This page is meant to be used as a way to mathematically explain the inner workings of machine learning. Through linear algebra, partial differential equations, and summations we work our way foward into understanding the basics behind backpropogation and neural networks.

All information was taken from this website: http://neuralnetworksanddeeplearning.com/index.html </br>
However, examples and concepts are expanded upon on this page.

Feel free to contact me if there are any issues.

## 2 - Perceptrons and Neurons

## 2.1 - Perceptrons

To first learn about machine learning, we have to learn about the **perceptron**. A perceptron takes in several binary (zero or one) inputs and produces a single binary output.

<p align="center">
  <img alt="perceptron" src="http://neuralnetworksanddeeplearning.com/images/tikz0.png" />
</p>

Here, in this example, we have a perceptron that takes three inputs: <img alt="x1" src="https://latex.codecogs.com/svg.latex?x_1" />, <img alt="x2" src="https://latex.codecogs.com/svg.latex?x_2" />, and <img alt="x3" src="https://latex.codecogs.com/svg.latex?x_3" />. It also has one output. The perceptron attaches weights to each input to express their realtive importance. For the example above, we would then have one weight for each of the inputs: <img alt="w1" src="https://latex.codecogs.com/svg.latex?w_1" />, <img alt="w2" src="https://latex.codecogs.com/svg.latex?w_2" />, and <img alt="w3" src="https://latex.codecogs.com/svg.latex?w_3" />. 

The output is determined by the weighted sum 

<p align="center">
  <img alt="weighted sum" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\sum_{k=1}^pw_kx_k" />
</p>

and whether this sum is less than or greater than an arbitrary thershold value (we will get into this soon). Here, in the equation above, the variable <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> represents the input and weight in question. The variable <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> represents the the number of inputs for that specific perceptron. Ordering of inputs is from top to bottom where <img alt="k=1" src="https://latex.codecogs.com/svg.latex?k=1" /> is the first input (topmost) and <img alt="k=p" src="https://latex.codecogs.com/svg.latex?k=p" /> is the last input (bottommost).

Putting it all together with the threshold, we can write this mathematically as

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{{\text{output}=\begin{cases}0\hspace{1cm}&\displaystyle{\sum_{k=1}^pw_kx_k\leq\text{threshold}}\\\\1&\displaystyle{\sum_{k=1}^pw_kx_k>\text{threshold}}\end{cases}}}" />
</p>

We can move the threshold to the other side of the inequality as shown

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{{\text{output}=\begin{cases}0\hspace{1cm}&\displaystyle{\sum_{k=1}^pw_kx_k-\text{threshold}\leq0}\\\\1&\displaystyle{\sum_{k=1}^pw_kx_k-\text{threshold}>0}\end{cases}}}" />
</p>

Then, we can define the threshold as the perceptron's bias

<p align="center">
  <img alt="b=-threshold" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{b=-\text{threshold}}" />
</p>

Finally, we have

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{{\text{output}=\begin{cases}0\hspace{1cm}&\displaystyle{\sum_{k=1}^pw_kx_k&plus;b\leq0}\\\\1&\displaystyle{\sum_{k=1}^pw_kx_k&plus;b>0}\end{cases}}}" />
</p>

### 2.1.1 Matrix Notation

For this next example, let us say we have a perceptron that takes in more than three inputs (<img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />). To simplify our summation, we can write this as a product between two matrices

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\sum_{k=1}^pw_kx_k&=w_1x_1&plus;w_2x_2&plus;w_3x_3&plus;\ldots&plus;w_px_p\\&=\begin{bmatrix}w_1&w_2&w_3&\cdots&w_p\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\\\vdots\\x_p\end{bmatrix}\\&=\mathbf{w}\mathbf{x}\end{align*}" />
</p>

Thus, we can simplify the output to our perceptron as

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{cases}0\hspace{1cm}&\mathbf{w}\mathbf{x}&plus;b\leq0\\\\1&\mathbf{w}\mathbf{x}&plus;b>0\end{cases}" />
</p>

### 2.1.2 - Problem Set

1. One about writing down all weights inputs for a 4-input system
2. One about "should I go biking today?" with factors such as weather, time of day, bike maintenance, "did I shower already?"
3. Last one about simplifying the weights and biases by dividing by the threshold, b

## 2.2 - Sigmoid Neurons

Instead of a perceptron, let us now consider a sigmoid neuron. To understand this, let us delve into the sigmoid function which is defined as


<p align="center">
  <img alt="sigmoid function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\sigma(z)=\frac{1}{1&plus;e^{-z}}" />
</p>

where <img alt="z" src="https://latex.codecogs.com/svg.latex?z" /> is the argument of our function.

Like the perceptron, we can define the output to our sigmoid neuron to be this sigmoid function rather than strictly binary. Writing this down, we have

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\text{output}&=\sigma\left(\sum_{k=1}^pw_kx_k&plus;b\right)\\&=\frac{1}{1&plus;e^{-(\sum_{k=1}^pw_kx_k&plus;b)}}\end{align*}" />
</p>
 
### 2.2.1 - Matrix Notation

Also like the perceptron, the output of a sigmoid neuron can be simplified using matrix notation. The output can be written as

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\text{output}&=\sigma(\mathbf{w}\mathbf{x}&plus;b)\\&=\frac{1}{1&plus;e^{-(\mathbf{w}\mathbf{x}&plus;b)}}\\\end{align*}" />
</p>

### 2.2.2 - Problem Set

1. Plot graph, show what happens as we approach z=1 or z=0
2. Previous problem set, but with sigmoid neurons
3. Take derivative of the sigmoid function

## 3 Neural Networks

## 3.1 Multi-layer Neural Networks

Rather than one neuron, let us consider a multiple-layer neural network.

<p align="center">
  <img alt="3layerneuralnetwork" src="http://neuralnetworksanddeeplearning.com/images/tikz16.png" />
</p>

Let us use the notation <img alt="w_(jk)^l" src="https://latex.codecogs.com/svg.latex?w_{jk}^l" /> to denote the weight for the connection of the <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> neuron in the previous layer, <img alt="l-1" src="https://latex.codecogs.com/svg.latex?l-1" />, to the <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neuron in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer. Writing this more explicitly, we have

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{a_j^l=\sigma\left(\sum_{k=1}^pw_{jk}^la_k^{l-1}&plus;b_j^l\right)}" />
</p>

where <img alt="a_j^l" src="https://latex.codecogs.com/svg.latex?a_j^l" /> is the output, or activation, and <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l-1" /> layer. Recall, for using this equation, <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> and <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> must be constants.

For more simplicity (will come in handy later), we can define the sigmoid argument as

<p align="center">
  <img alt="z" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{z_j^l=\sum_{k=1}^pw_{jk}^la_k^{l-1}&plus;b_j^l}" />
</p>

and

<p align="center">
  <img alt="z" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{a_j^l=\sigma(z_j^l)}" />
</p>

### 3.1.1 Matrix Notation

This can be written in matrix format for a constant <img alt="l" src="https://latex.codecogs.com/svg.latex?l" />. For this example, let us say the layer in question, <img alt="l" src="https://latex.codecogs.com/svg.latex?l" />, has more than three neurons, and thus had more than three activations (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" />). We define the activation matrix as

<p align="center">
  <img alt="activation matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{a}^l=\begin{bmatrix}a_1^l\\a_2^l\\a_3^l\\\vdots\\a_m^l\end{bmatrix}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer.

Let us also say, for this example, that the previous layer, <img alt="l-1" src="https://latex.codecogs.com/svg.latex?l-1" />, has more than three neurons (<img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />). We can define the activations from the previous layer as

<p align="center">
  <img alt="previous layer activation matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{a}^{l-1}=\begin{bmatrix}a_1^{l-1}\\a_2^{l-1}\\a_3^{l-1}\\\vdots\\a_p^{l-1}\end{bmatrix}" />
</p>

where <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of neurons in the <img alt="l-1" src="https://latex.codecogs.com/svg.latex?l-1" /> layer.

Because the layer in question,<img alt="l" src="https://latex.codecogs.com/svg.latex?l" />, has more than three neurons (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" />), we can also define the bias matrix as

<p align="center">
  <img alt="bias matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{b}^l=\begin{bmatrix}b_1^l\\b_2^l\\b_3^l\\\vdots\\b_m^l\end{bmatrix}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer.

Finally, we can define the weight matrix (recalling that (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" /> and <img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />) as

<p align="center">
  <img alt="weight matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{w}^l=\begin{bmatrix}w_{11}^l&w_{12}^l&w_{13}^l&\hdots&w_{1p}^l\\w_{21}^l&w_{22}^l&w_{23}^l&\hdots&w_{2p}^l\\w_{31}^l&w_{32}^l&w_{33}^l&\hdots&w_{3p}^l\\\vdots&\vdots&\vdots&\ddots&\vdots\\w_{m1}^l&w_{m2}^l&w_{m3}^l&\hdots&w_{mp}^l\end{bmatrix}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer and <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of neurons in the <img alt="l-1" src="https://latex.codecogs.com/svg.latex?l-1" /> layer.

Putting everything together, we have

<p align="center">
  <img alt="formula" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\begin{bmatrix}a_1^l\\a_2^l\\a_3^l\\\vdots\\a_m^l\end{bmatrix}&=\sigma\left(\begin{bmatrix}w_{11}^l&w_{12}^l&w_{13}^l&\hdots&w_{1p}^l\\w_{21}^l&w_{22}^l&w_{23}^l&\hdots&w_{2p}^l\\w_{31}^l&w_{32}^l&w_{33}^l&\hdots&w_{3p}^l\\\vdots&\vdots&\vdots&\ddots&\vdots\\w_{m1}^l&w_{m2}^l&w_{m3}^l&\hdots&w_{mp}^l\end{bmatrix}\begin{bmatrix}a_1^{l-1}\\a_2^{l-1}\\a_3^{l-1}\\\vdots\\a_p^{l-1}\end{bmatrix}&plus;\begin{bmatrix}b_1^l\\b_2^l\\b_3^l\\\vdots\\b_m^l\end{bmatrix}\right)\\\mathbf{a}^l&=\sigma(\mathbf{w}^l\mathbf{a}^{l-1}&plus;\mathbf{b}^l)\end{align*}" />
</p>

For more simplicity (will come in handy later), we can define the sigmoid argument as

<p align="center">
  <img alt="z" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{z}^l=\mathbf{w}^l\mathbf{a}^{l-1}&plus;\mathbf{b}^l" />
</p>

and is defined liked the matrices before as

<p align="center">
  <img alt="argument matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{z}^l=\begin{bmatrix}z_1^l\\z_2^l\\z_3^l\\\vdots\\z_m^l\end{bmatrix}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer.

Likewise, we can also say

<p align="center">
  <img alt="formulas" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\mathbf{a}^{l}=\sigma(\mathbf{z}^{l})" />
</p>


### 3.1.2 - Problem Set

1. Example number a multi-layer nueral network
2. Example multiplying stuff out using summations vs matrix
3. Number problem

## 3.2 - Quadratic Cost Function

Let us say we want to see how well our neural network is doing. We would then need to feed our neural network examples that we know the output to. We would then need to compare it to the real output of our neural network and compare how off we are. To properly calculate how well our neural network is doing (and to properly get weights and biases), we must define a cost function. We define the quadratic cost function as

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{n}\sum_{x=1}^n\left(\frac{1}{2}\sum_{j=1}^m\left(y_j(x)-a_j^L(x)\right)^2\right)}" />
</p>

where <img alt="n" src="https://latex.codecogs.com/svg.latex?n" /> is the total number of training examples, <img alt="x" src="https://latex.codecogs.com/svg.latex?x" /> is the individual training example or training example in question, <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is number of neurons in the last layer <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />, <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> is the variable that designates the activation of the training example and output of our neuron in question, <img alt="y_j(x)" src="https://latex.codecogs.com/svg.latex?y_j(x)" /> is the desired output of the nueron in question from the number training example <img alt="x" src="https://latex.codecogs.com/svg.latex?x" />, and <img alt="a_j^L(x)" src="https://latex.codecogs.com/svg.latex?a_j^L(x)" /> is the actual output from of the nueron in question from the number training example <img alt="x" src="https://latex.codecogs.com/svg.latex?x" />.

For simplicty, once again, we can define

<p align="center">
  <img alt="C_x" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C_x=\frac{1}{2}\sum_{j=1}^m\left(y_j(x)-a_j^L(x)\right)^2}" />
</p>

and thus we can also say

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{n}\sum_{x=1}^nC_x}" />
</p>


### 3.2.1 - Matrix Notation

Like before, we can attempt to write this in matrix notation using examples. For example, let us sat that we have more than one training examples (<img alt="n greater than 1" src="https://latex.codecogs.com/svg.latex?n>1" />), and our last layer has more than three activations (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" />). Our training examples matrices will look like so

<p align="center">
  <img alt="training examples matrices" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\mathbf{y}(1)&=\begin{bmatrix}y_1(1)\\y_2(1)\\y_3(1)\\\vdots\\y_m(1)\end{bmatrix}\\\vdots\\\mathbf{y}(n)&=\begin{bmatrix}y_1(n)\\y_2(n)\\y_3(n)\\\vdots\\y_m(n)\end{bmatrix}\end{align*}" />
</p>

where <img alt="n" src="https://latex.codecogs.com/svg.latex?n" /> is the total number of training examples and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

We can write the activations of our last layer in a similar fashion as

<p align="center">
  <img alt="activations matrices" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\mathbf{a}^L(1)&=\begin{bmatrix}a_1^L(1)\\a_2^L(1)\\a_3^L(1)\\\vdots\\a_m^L(1)\end{bmatrix}\\\vdots\\\mathbf{a}^L(n)&=\begin{bmatrix}a_1^L(n)\\a_2^L(n)\\a_3^L(n)\\\vdots\\a_m^L(n)\end{bmatrix}\end{align*}" />
</p>

where <img alt="n" src="https://latex.codecogs.com/svg.latex?n" /> is the total number of training examples and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

We get an equivalent statement to that of the summation notation. Writing this out using our examples above, we have

<p align="center">
  <img alt="quadratic cost function matrix example" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\left\|\mathbf{y}(1)-\mathbf{a}^L(1)\right\|^2&=\left(\sqrt{(\mathbf{y}(1)-\mathbf{a}^L(1))\cdot(\mathbf{y}(1)-\mathbf{a}^L(1))}\right)^2\\&=(\mathbf{y}(1)-\mathbf{a}^L(1))\cdot(\mathbf{y}(1)-\mathbf{a}^L(1))\\&=\begin{bmatrix}y_1(1)-a_1^L(1)\\y_2(1)-a_2^L(1)\\y_3(1)-a_3^L(1)\\\vdots\\y_m(1)-a_m^L(1)\\\end{bmatrix}^T\begin{bmatrix}y_1(1)-a_1^L(1)\\y_2(1)-a_2^L(1)\\y_3(1)-a_3^L(1)\\\vdots\\y_m(1)-a_m^L(1)\\\end{bmatrix}\\&=\left(y_1(1)-a_1^L(1)\right)^2&plus;\left(y_2(1)-a_2^L(1)\right)^2&plus;\left(y_3(1)-a_3^L(1)\right)^2&plus;\hdots&plus;\left(y_m(1)-a_m^L(1)\right)^2\\&=\sum_{j=1}^m\left(y_j(1)-a_j^L(1)\right)^2\end{align*}" />
</p>

and for the general case we have

<p align="center">
  <img alt="quadratic cost function matrix general" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}\left\|\mathbf{y}(n)-\mathbf{a}^L(n)\right\|^2&=\left(\sqrt{(\mathbf{y}(n)-\mathbf{a}^L(n))\cdot(\mathbf{y}(n)-\mathbf{a}^L(n))}\right)^2\\&=(\mathbf{y}(n)-\mathbf{a}^L(n))\cdot(\mathbf{y}(n)-\mathbf{a}^L(n))\\&=\begin{bmatrix}y_1(n)-a_1^L(n)\\y_2(n)-a_2^L(n)\\y_3(n)-a_3^L(n)\\\vdots\\y_m(n)-a_m^L(n)\\\end{bmatrix}^T\begin{bmatrix}y_1(n)-a_1^L(n)\\y_2(n)-a_2^L(n)\\y_3(n)-a_3^L(n)\\\vdots\\y_m(n)-a_m^L(n)\\\end{bmatrix}\\&=\left(y_1(n)-a_1^L(n)\right)^2&plus;\left(y_2(n)-a_2^L(n)\right)^2&plus;\left(y_3(n)-a_3^L(n)\right)^2&plus;\hdots&plus;\left(y_m(n)-a_m^L(n)\right)^2\\&=\sum_{j=1}^m\left(y_j(n)-a_j^L(n)\right)^2\end{align*}" />
</p>

where <img alt="n" src="https://latex.codecogs.com/svg.latex?n" /> is the total number of training examples and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Thus, we can say that

<p align="center">
  <img alt="C_x" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\begin{align*}C_x&=\frac{1}{2}\sum_{j=1}^m\left(y_j(x)-a_j^L(x)\right)^2\\&=\frac{1}{2}\left\|\mathbf{y}(x)-\mathbf{a}^L(x)\right\|^2\end{align*}" />
</p>

Since we know that

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{n}\sum_{x=1}^nC_x}" />
</p>

we finally have

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{n}\sum_{x=1}^n\left(\frac{1}{2}\left\|\mathbf{y}(x)-\mathbf{a}^L(x)\right\|^2\right)}" />
</p>

where <img alt="x" src="https://latex.codecogs.com/svg.latex?x" /> is the individual training example or training example in question, <img alt="n" src="https://latex.codecogs.com/svg.latex?n" /> is the max number of training examples, <img alt="y(x) matrix" src="https://latex.codecogs.com/svg.latex?\mathbf{y}(x)" /> is the desired output matrix when training example <img alt="x" src="https://latex.codecogs.com/svg.latex?x" /> is inputted to the network, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" /> is the last layer of our neural network, and <img alt="a^L(x) matrix" src="https://latex.codecogs.com/svg.latex?\mathbf{a}^L(x)" /> is the matrix of activations from the last layer of our neural network when training example <img alt="x" src="https://latex.codecogs.com/svg.latex?x" /> is inputted to the network.

### 3.2.2 - Problem Set
1. Number problems using summations and matrices

## 4 - Four Fundamental Equations of Back Backpropagation

## 4.1 - Error

Say we want to know how our system would change if there were a change in output from any given neuron. How would we do that? We would do it using partial differential equations. We define error of the the <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neuron in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer as

<p align="center">
  <img alt="error_j^l" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\delta_j^l=\frac{\partial{C}}{\partial{z_j^l}}" />
</p>

We will use the variable alot to relate to other quantities using backpropagation.  

## 4.2 - First Fundamental Equation: An Equation for the Error in the Output Layer

Let us say that for this next example, we only have one training example. As a result, the cost function can be written as

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\sum_{j=1}^m\left(y_j-a_j^L\right)^2}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />. We drop the <img alt="x" src="https://latex.codecogs.com/svg.latex?x" /> notation given that there is only one training example.

Let us say our last layer has more than three nuerons (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" />). Explicitly writing out the cost function, we have

<p align="center">
  <img alt="quadratic cost function example" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\left((y_1-a_1^L)^2&plus;(y_2-a_2^L)^2&plus;(y_3-a_3^L)^2&plus;\hdots&plus;(y_m-a_m^L)^2\right)}" />
</p>

Using the chain rule to find <img alt="partial C" src="https://latex.codecogs.com/svg.latex?\partial{C}" /> we have

<p align="center">
  <img alt="partial C" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{a_1^L}}\partial{a_1^L}&plus;\frac{\partial{C}}{\partial{a_2^L}}\partial{a_2^L}&plus;\frac{\partial{C}}{\partial{a_3^L}}\partial{a_3^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{a_m^L}}\partial{a_m^L}}" />
</p>

Recall the relationship between <img alt="a_j^l" src="https://latex.codecogs.com/svg.latex?a_j^l" /> and <img alt="z_j^l" src="https://latex.codecogs.com/svg.latex?z_j^l" />

<p align="center">
  <img alt="a_j^l = sigma z_j^l" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}a_j^l&=\sigma(z_j^l)\\a_j^L&=\sigma(z_j^L)\end{align*}}" />
</p>
  
Plugging in the numbers specific to our example, we have
  
<p align="center">
  <img alt="a_j^l = sigma z_j^l example" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}a_1^L=&\sigma(z_1^L)\\a_2^L=&\sigma(z_2^L)\\a_3^L=&\sigma(z_3^L)\\&\vdots\\a_m^L=&\sigma(z_m^L)\end{align*}}" />
</p>

Using the chain rule once again to find <img alt="partial a_1^L" src="https://latex.codecogs.com/svg.latex?\partial{a_1^L}" />

<p align="center">
  <img alt="partial a_j^l examples" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{a}_1^L=&\frac{\partial{a}_1^L}{\partial{z}_1^L}\partial{z}_1^L\\\partial{a}_2^L=&\frac{\partial{a}_2^L}{\partial{z}_2^L}\partial{z}_2^L\\\partial{a}_3^L=&\frac{\partial{a}_3^L}{\partial{z}_3^L}\partial{z}_3^L\\&\vdots\\\partial{a}_m^L=&\frac{\partial{a}_m^L}{\partial{z}_m^L}\partial{z}_m^L\end{align*}}" />
</p>

Substituting our partials in, we get

<p align="center">
  <img alt="partial C expanded 1 example" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{a_1^L}}\frac{\partial{a}_1^L}{\partial{z}_1^L}\partial{z}_1^L&plus;\frac{\partial{C}}{\partial{a_2^L}}\frac{\partial{a}_2^L}{\partial{z}_2^L}\partial{z}_2^L&plus;\frac{\partial{C}}{\partial{a_3^L}}\frac{\partial{a}_3^L}{\partial{z}_3^L}\partial{z}_3^L&plus;\hdots&plus;\frac{\partial{C}}{\partial{a_m^L}}\frac{\partial{a}_m^L}{\partial{z}_m^L}\partial{z}_m^L}" />
</p>

Also recall that our cost function can we written as

<p align="center">
  <img alt="cost function wrt to sigma z" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\sum_{j=1}^m\left(y_j-\sigma(z_j^L)\right)^2}" />
</p>

and thus, written out for this example, we have


<p align="center">
  <img alt="cost function wrt to sigma z example" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\left((y_1-\sigma(z_1^L))^2&plus;(y_2-\sigma(z_2^L))^2&plus;(y_3-\sigma(z_3^L))^2&plus;\hdots&plus;(y_m-\sigma(z_m^L))^2\right)}" />
</p>

Evaluating <img alt="partial C" src="https://latex.codecogs.com/svg.latex?\partial{C}" /> for this as well, we have

<p align="center">
  <img alt="partial C expanded 2 example" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{z_1^L}}\partial{z_1^L}&plus;\frac{\partial{C}}{\partial{z_2^L}}\partial{z_2^L}&plus;\frac{\partial{C}}{\partial{z_3^L}}\partial{z_3^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^L}}\partial{z_m^L}}" />


Equating both equations, we get

<p align="center">
  <img alt="both partial C examples equated" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\frac{\partial{C}}{\partial{z_1^L}}\partial{z_1^L}&plus;\frac{\partial{C}}{\partial{z_2^L}}\partial{z_2^L}&plus;\frac{\partial{C}}{\partial{z_3^L}}\partial{z_3^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^L}}\partial{z_m^L}=\frac{\partial{C}}{\partial{a_1^L}}\frac{\partial{a}_1^L}{\partial{z}_1^L}\partial{z}_1^L&plus;\frac{\partial{C}}{\partial{a_2^L}}\frac{\partial{a}_2^L}{\partial{z}_2^L}\partial{z}_2^L&plus;\frac{\partial{C}}{\partial{a_3^L}}\frac{\partial{a}_3^L}{\partial{z}_3^L}\partial{z}_3^L&plus;\hdots&plus;\frac{\partial{C}}{\partial{a_m^L}}\frac{\partial{a}_m^L}{\partial{z}_m^L}\partial{z}_m^L}" />
</p>  

Grouping like terms, we can see that

<p align="center">
  <img alt="like terms" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\frac{\partial{C}}{\partial{z_1^L}}\partial{z_1^L}=&\frac{\partial{C}}{\partial{a_1^L}}\frac{\partial{a}_1^L}{\partial{z}_1^L}\partial{z}_1^L\\\frac{\partial{C}}{\partial{z_2^L}}\partial{z_2^L}=&\frac{\partial{C}}{\partial{a_2^L}}\frac{\partial{a}_2^L}{\partial{z}_2^L}\partial{z}_2^L\\\frac{\partial{C}}{\partial{z_3^L}}\partial{z_3^L}=&\frac{\partial{C}}{\partial{a_3^L}}\frac{\partial{a}_3^L}{\partial{z}_3^L}\partial{z}_3^L\\&\vdots\\\frac{\partial{C}}{\partial{z_m^L}}\partial{z_m^L}=&\frac{\partial{C}}{\partial{a_m^L}}\frac{\partial{a}_m^L}{\partial{z}_m^L}\partial{z}_m^L&space;\end{align*}}" />
</p>

Recall the definition of the error

<p align="center">
  <img alt="error_j^l" src="https://latex.codecogs.com/svg.latex?\large\displaystyle\delta_j^l=\frac{\partial{C}}{\partial{z_j^l}}" />
</p>

Substituting and making more general, we have

<p align="center">
  <img alt="equation 1 of backpropagation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\delta^L_j&=\frac{\partial{C}}{\partial{a_j^L}}\frac{\partial{a_j^L}}{\partial{z_j^L}}\\&=\frac{\partial{C}}{\partial{a_j^L}}\sigma'(z_j^L)\end{align*}}" />
</p>

where <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> is a neuron in the last layer <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

## 4.2.1 - Matrix Notation

We can also write this in matrix format. Take, for example, a situation where we have atleast 3 neurons in the last layer <img alt="L" src="https://latex.codecogs.com/svg.latex?L" /> (<img alt="j greater than 3" src="https://latex.codecogs.com/svg.latex?j>3" />). We can define error as

<p align="center">
  <img alt="error matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\delta}^L=\begin{bmatrix}\delta_1^L\\\delta_2^L\\\delta_3^L\\\vdots\\\delta_m^L\end{bmatrix}}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Likewise, we can define the partial differentiation of our cost function with respect to our last layer's neuron input as

<p align="center">
  <img alt="partial C wrt to a matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\nabla}_aC=\begin{bmatrix}\frac{\partial{C}}{\partial{a_1^L}}\\\frac{\partial{C}}{\partial{a_2^L}}\\\frac{\partial{C}}{\partial{a_3^L}}\\\vdots\\\frac{\partial{C}}{\partial{a_m^L}}\end{bmatrix}}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Finally, we can define the derivative of our sigmoid function as

<p align="center">
  <img alt="sigmoid derivative matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\sigma}'(z^L)=\begin{bmatrix}\sigma'(z_1^l)\\\sigma'(z_2^L)\\\sigma'(z_3^L)\\\vdots\\\sigma'(z_m^L)\end{bmatrix}}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Putting it all together, we get

<p align="center">
  <img alt="error matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{bmatrix}\delta_1^L\\\delta_2^L\\\delta_3^L\\\vdots\\\delta_m^L\end{bmatrix}=\begin{bmatrix}\frac{\partial{C}}{\partial{a_1^L}}\\\frac{\partial{C}}{\partial{a_2^L}}\\\frac{\partial{C}}{\partial{a_3^L}}\\\vdots\\\frac{\partial{C}}{\partial{a_m^L}}\end{bmatrix}\odot\begin{bmatrix}\sigma'(z_1^l)\\\sigma'(z_2^L)\\\sigma'(z_3^L)\\\vdots\\\sigma'(z_m^L)\end{bmatrix}}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" /> and <img alt="Hadamard symbol" src="https://latex.codecogs.com/svg.latex?\odot" /> is the Hadamard product.

Simplifying, we get

<p align="center">
  <img alt="equation 1 of backpropagation matrix notation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\large\displaystyle{\boldsymbol{\delta}^L=\boldsymbol{\nabla}_aC\odot\boldsymbol{\sigma}'(z^L)}}" />
</p>

Specifically, for our quadratic cost function, we can write

<p align="center">
  <img alt="equation 1 of backpropagation matrix notation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\delta}^L=(\mathbf{a}^L-\mathbf{y})\odot\boldsymbol{\sigma}'(z^L)}" />
</p>

## 4.3 - Second Fundamental Equation: An Equation for the Error in Terms of Error in the Next Layer

Let us say that for this next example, we only have one training example. As a result, the cost function can be written as

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\sum_{j=1}^m\left(y_j-a_j^L\right)^2}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the amount of neurons in the last layer, <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />. We drop the <img alt="x" src="https://latex.codecogs.com/svg.latex?x" /> notation given that there is only one training example.

Let us then define this next variable for convenience (we will see later on) 

<p align="center">
  <img alt="L equals l+1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{L=l+1}" />
</p>

So, our quadratic cost function is now

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\sum_{j=1}^m\left(y_j-a_j^{l+1}\right)^2}" />
</p>

Let us say, for this example, that we have atleast 3 neurons in the last layer <img alt="l+1" src="https://latex.codecogs.com/svg.latex?l+1" /> (<img alt="j greater than 3" src="https://latex.codecogs.com/svg.latex?j>3" />). Writing our quadratic cost function more explicitly, we have

<p align="center">
  <img alt="quadratic cost function explicit" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\left[(y_1-a_1^{l&plus;1})^2&plus;(y_2-a_2^{l&plus;1})^2&plus;(y_3-a_3^{l&plus;1})^2]&plus;\hdots&plus;(y_m-a_m^{l&plus;1})^2\right]}" />
</p>

Recalling an earlier equation for the sigmoid argument, we have

<p align="center">
  <img alt="sigmoid argument l+1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}z_j^L&=\sum_{k=1}^pw_{jk}^La_k^{L-1}&plus;b_j^L\\z_j^{l&plus;1}&=\sum_{k=1}^pw_{jk}^{l&plus;1}a_k^{l}&plus;b_j^{l&plus;1}\end{align*}}" />
</p>

Plugging our sigmoid argument into our explicit quadratic cost function, we get

<p align="center">
  <img alt="quadratic cost function more explicit" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\left[(y_1-\sigma(z_1^{l&plus;1}))^2&plus;(y_2-\sigma(z_2^{l&plus;1}))^2&plus;(y_3-\sigma(z_3^{l&plus;1}))^2]&plus;\hdots&plus;(y_m-\sigma(z_m^{l&plus;1}))^2\right]}" />
</p>

Expanding even more, we get

<p align="center">
  <img alt="quadratic cost function with summations" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}C=\frac{1}{2}\Biggl[&\left(y_1-\sigma\left(\sum_{k=1}^pw_{1k}^{l&plus;1}a_k^{l}&plus;b_1^{l&plus;1}\right)\right)^2\\&plus;&\left(y_2-\sigma\left(\sum_{k=1}^pw_{2k}^{l&plus;1}a_k^{l}&plus;b_2^{l&plus;1}\right)\right)^2\\&plus;&\left(y_3-\sigma\left(\sum_{k=1}^pw_{3k}^{l&plus;1}a_k^{l}&plus;b_3^{l&plus;1}\right)\right)^2&plus;\hdots\\&plus;&\left(y_m-\sigma\left(\sum_{k=1}^pw_{mk}^{l&plus;1}a_k^{l}&plus;b_m^{l&plus;1}\right)\right)^2\Biggr]\end{align*}}" />
</p>

Let us say, for this example, that there are atleast 3 neurons in the second to last layer <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> (<img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />). Expandin our summations further, we get

<p align="center">
  <img alt="quadratic cost function with expanded summations" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}C=\frac{1}{2}\Bigl[&\left(y_1-\sigma\left(w_{11}^{l&plus;1}\sigma(z_1^l)&plus;w_{12}^{l&plus;1}\sigma(z_2^l)&plus;w_{13}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{1p}^{l&plus;1}\sigma(z_p^l)&plus;b_1^{l&plus;1}\right)\right)^2\\&plus;&\left(y_2-\sigma\left(w_{21}^{l&plus;1}\sigma(z_1^l)&plus;w_{22}^{l&plus;1}\sigma(z_2^l)&plus;w_{23}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{2p}^{l&plus;1}\sigma(z_p^l)&plus;b_2^{l&plus;1}\right)\right)^2\\&plus;&\left(y_3-\sigma\left(w_{31}^{l&plus;1}\sigma(z_1^l)&plus;w_{32}^{l&plus;1}\sigma(z_2^l)&plus;w_{33}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{3p}^{l&plus;1}\sigma(z_p^l)&plus;b_3^{l&plus;1}\right)\right)^2&plus;\hdots\\&plus;&\left(y_m-\sigma\left(w_{m1}^{l&plus;1}\sigma(z_1^l)&plus;w_{m2}^{l&plus;1}\sigma(z_2^l)&plus;w_{m3}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{mp}^{l&plus;1}\sigma(z_p^l)&plus;b_m^{l&plus;1}\right)\right)^2\Bigr]\end{align*}}" />
</p>

Taking the partial differentiation with respect to our cost function, we have

<p align="center">
  <img alt="quadratic cost function with expanded summations" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{z_1^l}}\partial{z_1^l}&plus;\frac{\partial{C}}{\partial{z_2^l}}\partial{z_2^l}&plus;\frac{\partial{C}}{\partial{z_3^l}}\partial{z_3^l}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_p^l}}\partial{z_p^l}&plus;\text{other&space;terms}}" />
</p>

where some terms are omitted to provide succinctness.

Recall the sigmoid arguments for layer <img alt="l+1" src="https://latex.codecogs.com/svg.latex?l+1" />, we have

<p align="center">
  <img alt="sigmoid arguments z l+1 examples" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}z_1^{l&plus;1}&=w_{11}^{l&plus;1}\sigma(z_1^l)&plus;w_{12}^{l&plus;1}\sigma(z_2^l)&plus;w_{13}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{1p}^{l&plus;1}\sigma(z_p^l)&plus;b_1^{l&plus;1}\\z_2^{l&plus;1}&=w_{21}^{l&plus;1}\sigma(z_1^l)&plus;w_{22}^{l&plus;1}\sigma(z_2^l)&plus;w_{23}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{2p}^{l&plus;1}\sigma(z_p^l)&plus;b_2^{l&plus;1}\\z_3^{l&plus;1}&=w_{31}^{l&plus;1}\sigma(z_1^l)&plus;w_{32}^{l&plus;1}\sigma(z_2^l)&plus;w_{33}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{3p}^{l&plus;1}\sigma(z_p^l)&plus;b_3^{l&plus;1}\\&\vdots\\z_m^{l&plus;1}&=w_{m1}^{l&plus;1}\sigma(z_1^l)&plus;w_{m2}^{l&plus;1}\sigma(z_2^l)&plus;w_{m3}^{l&plus;1}\sigma(z_3^l)&plus;\hdots&plus;w_{mp}^{l&plus;1}\sigma(z_p^l)&plus;b_m^{l&plus;1}\end{align*}}" />
</p>

Finding the partial differentiation for these equations, we have

<p align="center">
  <img alt="partials of sigmoid arguments z l+1 examples part 1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_1^{l&plus;1}}=&\frac{\partial{z_1^{l&plus;1}}}{\partial{z_1^l}}\partial{z_1^l}&plus;\frac{\partial{z_1^{l&plus;1}}}{\partial{z_2^l}}\partial{z_2^l}&plus;\frac{\partial{z_1^{l&plus;1}}}{\partial{z_3^l}}\partial{z_3^l}&plus;\hdots&plus;\frac{\partial{z_1^{l&plus;1}}}{\partial{z_p^l}}\partial{z_p^l}\\&plus;&\frac{\partial{z_1^{l&plus;1}}}{\partial{w_{11}^{l&plus;1}}}\partial{w_{11}^{l&plus;1}}&plus;\frac{\partial{z_1^{l&plus;1}}}{\partial{w_{12}^{l&plus;1}}}\partial{w_{12}^{l&plus;1}}&plus;\frac{\partial{z_1^{l&plus;1}}}{\partial{w_{13}^{l&plus;1}}}\partial{w_{13}^{l&plus;1}}&plus;\hdots&plus;\frac{\partial{z_1^{l&plus;1}}}{\partial{w_{1p}^{l&plus;1}}}\partial{w_{1p}^{l&plus;1}}\\&plus;&\frac{\partial{z_1^{l&plus;1}}}{\partial{b_1^{l&plus;1}}}\partial{b_1^{l&plus;1}}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z l+1 examples part 2" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_2^{l&plus;1}}=&\frac{\partial{z_2^{l&plus;1}}}{\partial{z_1^l}}\partial{z_1^l}&plus;\frac{\partial{z_2^{l&plus;1}}}{\partial{z_2^l}}\partial{z_2^l}&plus;\frac{\partial{z_2^{l&plus;1}}}{\partial{z_3^l}}\partial{z_3^l}&plus;\hdots&plus;\frac{\partial{z_2^{l&plus;1}}}{\partial{z_p^l}}\partial{z_p^l}\\&plus;&\frac{\partial{z_2^{l&plus;1}}}{\partial{w_{21}^{l&plus;1}}}\partial{w_{21}^{l&plus;1}}&plus;\frac{\partial{z_2^{l&plus;1}}}{\partial{w_{22}^{l&plus;1}}}\partial{w_{22}^{l&plus;1}}&plus;\frac{\partial{z_2^{l&plus;1}}}{\partial{w_{23}^{l&plus;1}}}\partial{w_{23}^{l&plus;1}}&plus;\hdots&plus;\frac{\partial{z_2^{l&plus;1}}}{\partial{w_{2p}^{l&plus;1}}}\partial{w_{2p}^{l&plus;1}}\\&plus;&\frac{\partial{z_2^{l&plus;1}}}{\partial{b_2^{l&plus;1}}}\partial{b_2^{l&plus;1}}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z l+1 examples part 3" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_3^{l&plus;1}}=&\frac{\partial{z_3^{l&plus;1}}}{\partial{z_1^l}}\partial{z_1^l}&plus;\frac{\partial{z_3^{l&plus;1}}}{\partial{z_2^l}}\partial{z_2^l}&plus;\frac{\partial{z_3^{l&plus;1}}}{\partial{z_3^l}}\partial{z_3^l}&plus;\hdots&plus;\frac{\partial{z_3^{l&plus;1}}}{\partial{z_p^l}}\partial{z_p^l}\\&plus;&\frac{\partial{z_3^{l&plus;1}}}{\partial{w_{31}^{l&plus;1}}}\partial{w_{31}^{l&plus;1}}&plus;\frac{\partial{z_3^{l&plus;1}}}{\partial{w_{32}^{l&plus;1}}}\partial{w_{32}^{l&plus;1}}&plus;\frac{\partial{z_3^{l&plus;1}}}{\partial{w_{33}^{l&plus;1}}}\partial{w_{33}^{l&plus;1}}&plus;\hdots&plus;\frac{\partial{z_3^{l&plus;1}}}{\partial{w_{3p}^{l&plus;1}}}\partial{w_{3p}^{l&plus;1}}\\&plus;&\frac{\partial{z_3^{l&plus;1}}}{\partial{b_3^{l&plus;1}}}\partial{b_3^{l&plus;1}}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z l+1 examples part m" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_m^{l&plus;1}}=&\frac{\partial{z_m^{l&plus;1}}}{\partial{z_1^l}}\partial{z_1^l}&plus;\frac{\partial{z_m^{l&plus;1}}}{\partial{z_2^l}}\partial{z_2^l}&plus;\frac{\partial{z_m^{l&plus;1}}}{\partial{z_3^l}}\partial{z_3^l}&plus;\hdots&plus;\frac{\partial{z_m^{l&plus;1}}}{\partial{z_p^l}}\partial{z_p^l}\\&plus;&\frac{\partial{z_m^{l&plus;1}}}{\partial{w_{m1}^{l&plus;1}}}\partial{w_{m1}^{l&plus;1}}&plus;\frac{\partial{z_m^{l&plus;1}}}{\partial{w_{m2}^{l&plus;1}}}\partial{w_{m2}^{l&plus;1}}&plus;\frac{\partial{z_m^{l&plus;1}}}{\partial{w_{m3}^{l&plus;1}}}\partial{w_{m3}^{l&plus;1}}&plus;\hdots&plus;\frac{\partial{z_m^{l&plus;1}}}{\partial{w_{mp}^{l&plus;1}}}\partial{w_{mp}^{l&plus;1}}\\&plus;&\frac{\partial{z_m^{l&plus;1}}}{\partial{b_m^{l&plus;1}}}\partial{b_m^{l&plus;1}}\end{align*}}" />
</p>

We then find the partial differentation of the cost function using the equation

<p align="center">
  <img alt="quadratic cost function more explicit" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{C=\frac{1}{2}\left[(y_1-\sigma(z_1^{l&plus;1}))^2&plus;(y_2-\sigma(z_2^{l&plus;1}))^2&plus;(y_3-\sigma(z_3^{l&plus;1}))^2]&plus;\hdots&plus;(y_m-\sigma(z_m^{l&plus;1}))^2\right]}" />
</p>

and we get

<p align="center">
  <img alt="quadratic cost function more explicit partial differentiation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\partial{z_1^{l&plus;1}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\partial{z_2^{l&plus;1}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\partial{z_3^{l&plus;1}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\partial{z_m^{l&plus;1}}}" />
</p>

We then substitute the equations above and group like terms (we omit some terms to also provide succinctness) to get

<p align="center">
  <img alt="grouped terms part 1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{C}=&\partial{z_1^l}\left(\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\frac{\partial{z_1^{l&plus;1}}}{\partial{z_1^l}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\frac{\partial{z_2^{l&plus;1}}}{\partial{z_1^l}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\frac{\partial{z_3^{l&plus;1}}}{\partial{z_1^l}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\frac{\partial{z_m^{l&plus;1}}}{\partial{z_1^l}}\right)\\&plus;&\partial{z_2^l}\left(\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\frac{\partial{z_1^{l&plus;1}}}{\partial{z_2^l}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\frac{\partial{z_2^{l&plus;1}}}{\partial{z_2^l}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\frac{\partial{z_3^{l&plus;1}}}{\partial{z_2^l}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\frac{\partial{z_m^{l&plus;1}}}{\partial{z_2^l}}\right)\end{align*}}" />
</p>
<p align="center">
  <img alt="grouped terms part 2" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}&plus;&\partial{z_3^l}\left(\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\frac{\partial{z_1^{l&plus;1}}}{\partial{z_3^l}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\frac{\partial{z_2^{l&plus;1}}}{\partial{z_3^l}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\frac{\partial{z_3^{l&plus;1}}}{\partial{z_3^l}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\frac{\partial{z_m^{l&plus;1}}}{\partial{z_3^l}}\right)\\&plus;&\hdots\\&plus;&\partial{z_p^l}\left(\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\frac{\partial{z_1^{l&plus;1}}}{\partial{z_p^l}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\frac{\partial{z_2^{l&plus;1}}}{\partial{z_p^l}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\frac{\partial{z_3^{l&plus;1}}}{\partial{z_p^l}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\frac{\partial{z_m^{l&plus;1}}}{\partial{z_p^l}}\right)\\&plus;&\text{other&space;terms}\end{align*}}" />
</p>

Equating this to our other partial differentiation equation, we can finally say that

<p align="center">
  <img alt="partials relationship" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\frac{\partial{C}}{\partial{z_1^l}}&=\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\frac{\partial{z_1^{l&plus;1}}}{\partial{z_1^l}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\frac{\partial{z_2^{l&plus;1}}}{\partial{z_1^l}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\frac{\partial{z_3^{l&plus;1}}}{\partial{z_1^l}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\frac{\partial{z_m^{l&plus;1}}}{\partial{z_1^l}}\\&\vdots\\\frac{\partial{C}}{\partial{z_p^l}}&=\frac{\partial{C}}{\partial{z_1^{l&plus;1}}}\frac{\partial{z_1^{l&plus;1}}}{\partial{z_p^l}}&plus;\frac{\partial{C}}{\partial{z_2^{l&plus;1}}}\frac{\partial{z_2^{l&plus;1}}}{\partial{z_p^l}}&plus;\frac{\partial{C}}{\partial{z_3^{l&plus;1}}}\frac{\partial{z_3^{l&plus;1}}}{\partial{z_p^l}}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^{l&plus;1}}}\frac{\partial{z_m^{l&plus;1}}}{\partial{z_p^l}}\end{align*}}" />
</p>

Making it more general, we have

<p align="center">
  <img alt="general relationship" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\frac{\partial{C}}{\partial{z_k^l}}=\sum_{j=1}^m\frac{\partial{C}}{\partial{z_j^{l&plus;1}}}\frac{\partial{z_j^{l&plus;1}}}{\partial{z_k^l}}}" />
</p>

<p align="center">
  <img alt="sigmoid argument l+1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{z_j^{l&plus;1}=\sum_{k=1}^pw_{jk}^{l&plus;1}a_k^{l}&plus;b_j^{l&plus;1}}" />
</p>

where <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neurons in the <img alt="l+1" src="https://latex.codecogs.com/svg.latex?l+1" />. This equations holds true for all layers!

This equation can be simplified even more

<p align="center">
  <img alt="second fundamental equation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\frac{\partial{C}}{\partial{z_k^l}}&=\sum_{j=1}^m\frac{\partial{C}}{\partial{z_j^{l&plus;1}}}\frac{\partial{z_j^{l&plus;1}}}{\partial{z_k^l}}\\\delta_k^l&=\sum_{j=1}^m\delta_j^{l&plus;1}\frac{\partial{z_j^{l&plus;1}}}{\partial{z_k^l}}\\&=\sum_{j=1}^mw_{jk}^{l&plus;1}\delta_j^{l&plus;1}\sigma'(z_k^l)\end{align*}}" />
</p>

where

<p align="center">
  <img alt="variable substitutions" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\delta_k^l&=\frac{\partial{C}}{\partial{z_k^l}}\\\delta_j^{l&plus;1}&=\frac{\partial{C}}{\partial{z_j^{l&plus;1}}}\\\frac{\partial{z_j^{l&plus;1}}}{\partial{z_k^l}}&=\frac{\partial{(w_{jk}^{l&plus;1}\sigma(z_k^l)&plus;b_j^{l&plus;1})}}{\partial{z_k^l}}\\&=w_{jk}^{l&plus;1}\sigma'(z_k^l)\end{align*}}" />
</p>

Recall that for <img alt="delta_k^l" src="https://latex.codecogs.com/svg.latex?\delta_k^l" />, we are looking at a constant <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> at a time, and thus there is no need to expand <img alt="partial z_j^(l+1)" src="https://latex.codecogs.com/svg.latex?\partial{z_j^{l&plus;1}}" /> using a summation.

## 4.3.1 - Matrix Notation

Like the other equations, equation two can be defined using matrix notation. Let us say, for this example, that we have atleast three neurons (<img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />) in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer. We can define the error in that layer as

<p align="center">
  <img alt="error in l layer" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\delta}^l=\begin{bmatrix}\delta_1^l\\\delta_2^l\\\delta_3^l\\\vdots\\\delta_p^l\end{bmatrix}}" />
</p>

Likewise, we can define the weight matrix to be

<p align="center">
  <img alt="weight matrix l" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\mathbf{w}^{l&plus;1}=\begin{bmatrix}w_{11}^{l&plus;1}&w_{12}^{l&plus;1}&w_{13}^{l&plus;1}&\hdots&w_{1p}^{l&plus;1}\\w_{21}^{l&plus;1}&w_{22}^{l&plus;1}&w_{23}^{l&plus;1}&\hdots&w_{2p}^{l&plus;1}\\w_{31}^{l&plus;1}&w_{32}^{l&plus;1}&w_{33}^{l&plus;1}&\hdots&w_{3p}^{l&plus;1}\\\vdots&\vdots&\vdots&\ddots&\vdots\\w_{m1}^{l&plus;1}&w_{m2}^{l&plus;1}&w_{m3}^{l&plus;1}&\hdots&w_{mp}^{l&plus;1}\end{bmatrix}}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of neurons in the <img alt="l+1" src="https://latex.codecogs.com/svg.latex?l+1" /> layer and <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer.

We define the error in the next layer as 

<p align="center">
  <img alt="error in l+1 layer" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\delta}^{l&plus;1}=\begin{bmatrix}\delta_1^{l&plus;1}\\\delta_2^{l&plus;1}\\\delta_3^{l&plus;1}\\\vdots\\\delta_m^{l&plus;1}\end{bmatrix}}" />
</p>

Lastly, we define the derivative of our sigmoid output as 

<p align="center">
  <img alt="sigma prime z_k^l matrix" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\sigma}'(z^l)=\begin{bmatrix}\sigma'(z_1^l)\\\sigma'(z_2^l)\\\sigma'(z_3^l)\\\vdots\\\sigma'(z_p^l)\end{bmatrix}}" />
</p>

Putting it all together, we get 

<p align="center">
  <img alt="fundamental equation two matrix format" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\begin{bmatrix}\delta_1^l\\\delta_2^l\\\delta_3^l\\\vdots\\\delta_p^l\end{bmatrix}&=\left(\begin{bmatrix}w_{11}^{l&plus;1}&w_{12}^{l&plus;1}&w_{13}^{l&plus;1}&\hdots&w_{1p}^{l&plus;1}\\w_{21}^{l&plus;1}&w_{22}^{l&plus;1}&w_{23}^{l&plus;1}&\hdots&w_{2p}^{l&plus;1}\\w_{31}^{l&plus;1}&w_{32}^{l&plus;1}&w_{33}^{l&plus;1}&\hdots&w_{3p}^{l&plus;1}\\\vdots&\vdots&\vdots&\ddots&\vdots\\w_{m1}^{l&plus;1}&w_{m2}^{l&plus;1}&w_{m3}^{l&plus;1}&\hdots&w_{mp}^{l&plus;1}\end{bmatrix}^T\begin{bmatrix}\delta_1^{l&plus;1}\\\delta_2^{l&plus;1}\\\delta_3^{l&plus;1}\\\vdots\\\delta_m^{l&plus;1}\end{bmatrix}\right)\odot\begin{bmatrix}\sigma'(z_1^l)\\\sigma'(z_2^l)\\\sigma'(z_3^l)\\\vdots\\\sigma'(z_m^l)\end{bmatrix}\\\boldsymbol{\delta}^l&=((\mathbf{w}^{l&plus;1})^T\boldsymbol{\delta}^{l&plus;1})\odot\boldsymbol{\sigma}'(z^l)\end{align*}}" />
</p>

where <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of neurons in the <img alt="l+1" src="https://latex.codecogs.com/svg.latex?l+1" /> layer and <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of neurons in the <img alt="l" src="https://latex.codecogs.com/svg.latex?l" /> layer. 

## 4.4 - Third Fundamental Equation: The Rate of Change of the Cost with Respect to Any Bias in the Network

Recall the expanded version of our cost function where have atleast 3 neurons in the last layer <img alt="L" src="https://latex.codecogs.com/svg.latex?L" /> (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" />)

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}C&=\frac{1}{2}\sum_{j=1}^m\left(y_j-a_j^L\right)^2\\&=\frac{1}{2}\left[(y_1-\sigma(z_1^L))^2&plus;(y_2-\sigma(z_2^L))^2&plus;(y_3-\sigma(z_3^L))^2]&plus;\hdots&plus;(y_m-\sigma(z_m^L))^2\right]\end{align*}}" />
</p>

Finding the partial differential equation to our cost function, we get

<p align="center">
  <img alt="partial quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{z_1^L}}\partial{z_1^L}&plus;\frac{\partial{C}}{\partial{z_2^L}}\partial{z_2^L}&plus;\frac{\partial{C}}{\partial{z_3^L}}\partial{z_3^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^L}}\partial{z_m^L}}" />
</p>
  
Finding the equations to our sigmoid argument where there are atleast 3 neurons in the second to last layer <img alt="L-1" src="https://latex.codecogs.com/svg.latex?L-1" /> (<img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />), we also get

<p align="center">
  <img alt="sigmoids L" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}z_1^L&=w_{11}^L\sigma(z_1^{L-1})+w_{12}^L\sigma(z_2^{L-1})+w_{13}^L\sigma(z_3^{L-1})+\hdots+w_{1p}^L\sigma(z_p^{L-1})+b_1^L\\z_2^L&=w_{21}^L\sigma(z_1^{L-1})+w_{22}^L\sigma(z_2^{L-1})+w_{23}^L\sigma(z_3^{L-1})+\hdots+w_{2p}^L\sigma(z_p^{L-1})+b_2^L\\z_3^L&=w_{31}^L\sigma(z_1^{L-1})+w_{32}^L\sigma(z_2^{L-1})+w_{33}^L\sigma(z_3^{L-1})+\hdots+w_{3p}^L\sigma(z_p^{L-1})+b_3^L\\&\vdots\\z_m^L&=w_{m1}^L\sigma(z_1^{L-1})+w_{m2}^L\sigma(z_2^{L-1})+w_{m3}^L\sigma(z_3^{L-1})+\hdots+w_{mp}^L\sigma(z_p^{L-1})+b_m^L\end{align*}}" />
</p>

where <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> neurons in the <img alt="L-1" src="https://latex.codecogs.com/svg.latex?L-1" /> layer and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neurons in the <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Finding the partial differentiation to these equations exactly as before, we have

<p align="center">
  <img alt="partials of sigmoid arguments z L examples part 1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_1^L}=&\frac{\partial{z_1^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_1^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_1^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_1^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_1^L}}{\partial{w_{11}^L}}\partial{w_{11}^L}&plus;\frac{\partial{z_1^L}}{\partial{w_{12}^L}}\partial{w_{12}^L}&plus;\frac{\partial{z_1^L}}{\partial{w_{13}^L}}\partial{w_{13}^L}&plus;\hdots&plus;\frac{\partial{z_1^L}}{\partial{w_{1p}^L}}\partial{w_{1p}^L}\\&plus;&\frac{\partial{z_1^L}}{\partial{b_1^L}}\partial{b_1^L}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z L examples part 2" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_2^L}=&\frac{\partial{z_2^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_2^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_2^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_2^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_2^L}}{\partial{w_{21}^L}}\partial{w_{21}^L}&plus;\frac{\partial{z_2^L}}{\partial{w_{22}^L}}\partial{w_{22}^L}&plus;\frac{\partial{z_2^L}}{\partial{w_{23}^L}}\partial{w_{23}^L}&plus;\hdots&plus;\frac{\partial{z_2^L}}{\partial{w_{2p}^L}}\partial{w_{2p}^L}\\&plus;&\frac{\partial{z_2^L}}{\partial{b_2^L}}\partial{b_2^L}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z L examples part 3" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_3^L}=&\frac{\partial{z_3^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_3^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_3^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_3^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_3^L}}{\partial{w_{31}^L}}\partial{w_{31}^L}&plus;\frac{\partial{z_3^L}}{\partial{w_{32}^L}}\partial{w_{32}^L}&plus;\frac{\partial{z_3^L}}{\partial{w_{33}^L}}\partial{w_{33}^L}&plus;\hdots&plus;\frac{\partial{z_3^L}}{\partial{w_{3p}^L}}\partial{w_{3p}^L}\\&plus;&\frac{\partial{z_3^L}}{\partial{b_3^L}}\partial{b_3^L}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z L examples part m" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_m^L}=&\frac{\partial{z_m^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_m^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_m^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_m^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_m^L}}{\partial{w_{m1}^L}}\partial{w_{m1}^L}&plus;\frac{\partial{z_m^L}}{\partial{w_{m2}^L}}\partial{w_{m2}^L}&plus;\frac{\partial{z_m^L}}{\partial{w_{m3}^L}}\partial{w_{m3}^L}&plus;\hdots&plus;\frac{\partial{z_m^L}}{\partial{w_{mp}^L}}\partial{w_{mp}^L}\\&plus;&\frac{\partial{z_m^L}}{\partial{b_m^L}}\partial{b_m^L}\end{align*}}" />
</p>

Substituting and omitting some terms for succinctness

<p align="center">
  <img alt="grouped b terms" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\partial{b_1^L}\left(\frac{\partial{C}}{\partial{z_1^L}}\frac{\partial{z_1^L}}{\partial{b_1^L}}\right)&plus;\partial{b_2^L}\left(\frac{\partial{C}}{\partial{z_2^L}}\frac{\partial{z_2^L}}{\partial{b_2^L}}\right)&plus;\partial{b_3^L}\left(\frac{\partial{C}}{\partial{z_3^L}}\frac{\partial{z_3^L}}{\partial{b_3^L}}\right)&plus;\hdots&plus;\partial{b_m^L}\left(\frac{\partial{C}}{\partial{z_m^L}}\frac{\partial{z_m^L}}{\partial{b_m^L}}\right)&plus;\text{other&space;terms}}" />
</p>

Also, recall that our cost function can be written as followed
  
<p align="center">
  <img alt="cost function expanded" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}C=&\frac{1}{2}\Bigl[\left(y_1-\sigma\left(w_{11}^L\sigma(z_1^{L-1})&plus;w_{12}^L\sigma(z_2^{L-1})&plus;w_{13}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{1p}^L\sigma(z_p^{L-1})&plus;b_1^L\right)\right)^2\\&&plus;\left(y_2-\sigma\left(w_{21}^L\sigma(z_1^{L-1})&plus;w_{22}^L\sigma(z_2^{L-1})&plus;w_{23}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{2p}^L\sigma(z_p^{L-1})&plus;b_2^L\right)\right)^2\\&&plus;\left(y_3-\sigma\left(w_{31}^L\sigma(z_1^{L-1})&plus;w_{32}^L\sigma(z_2^{L-1})&plus;w_{33}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{3p}^L\sigma(z_p^{L-1})&plus;b_3^L\right)\right)^2&plus;\hdots\\&&plus;\left(y_m-\sigma\left(w_{m1}^L\sigma(z_1^{L-1})&plus;w_{m2}^L\sigma(z_2^{L-1})&plus;w_{m3}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{mp}^L\sigma(z_p^{L-1})&plus;b_m^L\right)\right)^2\Bigr]\end{align*}}" />
</p>

where <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> neurons in the <img alt="L-1" src="https://latex.codecogs.com/svg.latex?L-1" /> layer and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neurons in the <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Finding the partial differentiation to this equation, we also get

<p align="center">
  <img alt="partial cost function expanded" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{b_1^L}}\partial{b_1^L}&plus;\frac{\partial{C}}{\partial{b_2^L}}\partial{b_2^L}&plus;\frac{\partial{C}}{\partial{b_3^L}}\partial{b_3^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{b_m^L}}\partial{b_m^L}&plus;\text{other&space;terms}}" />
</p>

Finally, after equating both equations, we can see that

<p align="center">
  <img alt="partials equated" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\frac{\partial{C}}{\partial{b_1^L}}&=\frac{\partial{C}}{\partial{z_1^L}}\frac{\partial{z_1^L}}{\partial{b_1^L}}\\\frac{\partial{C}}{\partial{b_2^L}}&=\frac{\partial{C}}{\partial{z_2^L}}\frac{\partial{z_2^L}}{\partial{b_2^L}}\\\frac{\partial{C}}{\partial{b_3^L}}&=\frac{\partial{C}}{\partial{z_3^L}}\frac{\partial{z_3^L}}{\partial{b_3^L}}\\&\vdots\\\frac{\partial{C}}{\partial{b_m^L}}&=\frac{\partial{C}}{\partial{z_m^L}}\frac{\partial{z_m^L}}{\partial{b_m^L}}\end{align*}}" />
</p>

Because the bias is always a constant, this equation can be simplified further to 

<p align="center">
  <img alt="partials equated simplified" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\frac{\partial{C}}{\partial{b_1^L}}&=\delta_1^L\\\frac{\partial{C}}{\partial{b_2^L}}&=\delta_2^L\\\frac{\partial{C}}{\partial{b_3^L}}&=\delta_3^L\\&\vdots\\\frac{\partial{C}}{\partial{b_m^L}}&=\delta_m^L\end{align*}}" />
</p>

This relationship holds true for all layers. Thus, our third fundamental equation is 

<p align="center">
  <img alt="third fundamental equation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\frac{\partial{C}}{\partial{b_j^l}}=\delta_j^l}" />
</p>

This equations holds true for all layers!

## 4.5 - Fourth Fundamental Equation: The Rate of Change of the Cost with Respect to Any Weight in the Network

Recall the expanded version of our cost function where have atleast 3 neurons in the last layer <img alt="L" src="https://latex.codecogs.com/svg.latex?L" /> (<img alt="m greater than 3" src="https://latex.codecogs.com/svg.latex?m>3" />)

<p align="center">
  <img alt="quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}C&=\frac{1}{2}\sum_{j=1}^m\left(y_j-a_j^L\right)^2\\&=\frac{1}{2}\left[(y_1-\sigma(z_1^L))^2&plus;(y_2-\sigma(z_2^L))^2&plus;(y_3-\sigma(z_3^L))^2]&plus;\hdots&plus;(y_m-\sigma(z_m^L))^2\right]\end{align*}}" />
</p>

Finding the partial differential equation to our cost function, we get

<p align="center">
  <img alt="partial quadratic cost function" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\partial{C}=\frac{\partial{C}}{\partial{z_1^L}}\partial{z_1^L}&plus;\frac{\partial{C}}{\partial{z_2^L}}\partial{z_2^L}&plus;\frac{\partial{C}}{\partial{z_3^L}}\partial{z_3^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{z_m^L}}\partial{z_m^L}}" />
</p>
  
Finding the equations to our sigmoid argument where there are atleast 3 neurons in the second to last layer <img alt="L-1" src="https://latex.codecogs.com/svg.latex?L-1" /> (<img alt="p greater than 3" src="https://latex.codecogs.com/svg.latex?p>3" />), we also get

<p align="center">
  <img alt="sigmoids L" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}z_1^L&=w_{11}^L\sigma(z_1^{L-1})+w_{12}^L\sigma(z_2^{L-1})+w_{13}^L\sigma(z_3^{L-1})+\hdots+w_{1p}^L\sigma(z_p^{L-1})+b_1^L\\z_2^L&=w_{21}^L\sigma(z_1^{L-1})+w_{22}^L\sigma(z_2^{L-1})+w_{23}^L\sigma(z_3^{L-1})+\hdots+w_{2p}^L\sigma(z_p^{L-1})+b_2^L\\z_3^L&=w_{31}^L\sigma(z_1^{L-1})+w_{32}^L\sigma(z_2^{L-1})+w_{33}^L\sigma(z_3^{L-1})+\hdots+w_{3p}^L\sigma(z_p^{L-1})+b_3^L\\&\vdots\\z_m^L&=w_{m1}^L\sigma(z_1^{L-1})+w_{m2}^L\sigma(z_2^{L-1})+w_{m3}^L\sigma(z_3^{L-1})+\hdots+w_{mp}^L\sigma(z_p^{L-1})+b_m^L\end{align*}}" />
</p>

where <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> neurons in the <img alt="L-1" src="https://latex.codecogs.com/svg.latex?L-1" /> layer and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neurons in the <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Finding the partial differentiation to these equations exactly as before, we have

<p align="center">
  <img alt="partials of sigmoid arguments z L examples part 1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_1^L}=&\frac{\partial{z_1^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_1^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_1^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_1^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_1^L}}{\partial{w_{11}^L}}\partial{w_{11}^L}&plus;\frac{\partial{z_1^L}}{\partial{w_{12}^L}}\partial{w_{12}^L}&plus;\frac{\partial{z_1^L}}{\partial{w_{13}^L}}\partial{w_{13}^L}&plus;\hdots&plus;\frac{\partial{z_1^L}}{\partial{w_{1p}^L}}\partial{w_{1p}^L}\\&plus;&\frac{\partial{z_1^L}}{\partial{b_1^L}}\partial{b_1^L}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z L examples part 2" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_2^L}=&\frac{\partial{z_2^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_2^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_2^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_2^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_2^L}}{\partial{w_{21}^L}}\partial{w_{21}^L}&plus;\frac{\partial{z_2^L}}{\partial{w_{22}^L}}\partial{w_{22}^L}&plus;\frac{\partial{z_2^L}}{\partial{w_{23}^L}}\partial{w_{23}^L}&plus;\hdots&plus;\frac{\partial{z_2^L}}{\partial{w_{2p}^L}}\partial{w_{2p}^L}\\&plus;&\frac{\partial{z_2^L}}{\partial{b_2^L}}\partial{b_2^L}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z L examples part 3" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_3^L}=&\frac{\partial{z_3^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_3^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_3^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_3^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_3^L}}{\partial{w_{31}^L}}\partial{w_{31}^L}&plus;\frac{\partial{z_3^L}}{\partial{w_{32}^L}}\partial{w_{32}^L}&plus;\frac{\partial{z_3^L}}{\partial{w_{33}^L}}\partial{w_{33}^L}&plus;\hdots&plus;\frac{\partial{z_3^L}}{\partial{w_{3p}^L}}\partial{w_{3p}^L}\\&plus;&\frac{\partial{z_3^L}}{\partial{b_3^L}}\partial{b_3^L}\end{align*}}" />
</p>
<p align="center">
  <img alt="partials of sigmoid arguments z L examples part m" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{z_m^L}=&\frac{\partial{z_m^L}}{\partial{z_1^{L-1}}}\partial{z_1^{L-1}}&plus;\frac{\partial{z_m^L}}{\partial{z_2^{L-1}}}\partial{z_2^{L-1}}&plus;\frac{\partial{z_m^L}}{\partial{z_3^{L-1}}}\partial{z_3^{L-1}}&plus;\hdots&plus;\frac{\partial{z_m^L}}{\partial{z_p^{L-1}}}\partial{z_p^{L-1}}\\&plus;&\frac{\partial{z_m^L}}{\partial{w_{m1}^L}}\partial{w_{m1}^L}&plus;\frac{\partial{z_m^L}}{\partial{w_{m2}^L}}\partial{w_{m2}^L}&plus;\frac{\partial{z_m^L}}{\partial{w_{m3}^L}}\partial{w_{m3}^L}&plus;\hdots&plus;\frac{\partial{z_m^L}}{\partial{w_{mp}^L}}\partial{w_{mp}^L}\\&plus;&\frac{\partial{z_m^L}}{\partial{b_m^L}}\partial{b_m^L}\end{align*}}" />
</p>

Substituting and omitting some terms for succinctness

<p align="center">
  <img alt="weight substitutions part 1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{C}=&\frac{\partial{C}}{\partial{z_1^L}}(\frac{\partial{z_1^L}}{\partial{w_{11}^L}}\partial{w_{11}^L}&plus;\frac{\partial{z_1^L}}{\partial{w_{12}^L}}\partial{w_{12}^L}&plus;\frac{\partial{z_1^L}}{\partial{w_{13}^L}}\partial{w_{13}^L}&plus;\hdots&plus;\frac{\partial{z_1^L}}{\partial{w_{1p}^L}}\partial{w_{1p}^L})\\&plus;&\frac{\partial{C}}{\partial{z_2^L}}(\frac{\partial{z_2^L}}{\partial{w_{21}^L}}\partial{w_{21}^L}&plus;\frac{\partial{z_2^L}}{\partial{w_{22}^L}}\partial{w_{22}^L}&plus;\frac{\partial{z_2^L}}{\partial{w_{23}^L}}\partial{w_{23}^L}&plus;\hdots&plus;\frac{\partial{z_2^L}}{\partial{w_{2p}^L}}\partial{w_{2p}^L})\\&plus;&\frac{\partial{C}}{\partial{z_3^L}}(\frac{\partial{z_3^L}}{\partial{w_{31}^L}}\partial{w_{31}^L}&plus;\frac{\partial{z_3^L}}{\partial{w_{32}^L}}\partial{w_{32}^L}&plus;\frac{\partial{z_3^L}}{\partial{w_{33}^L}}\partial{w_{33}^L}&plus;\hdots&plus;\frac{\partial{z_3^L}}{\partial{w_{3p}^L}}\partial{w_{3p}^L})&plus;\hdots\end{align*}}" />
</p>
<p align="center">
  <img alt="weight substitutions part 2" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{&plus;\frac{\partial{C}}{\partial{z_m^L}}(\frac{\partial{z_m^L}}{\partial{w_{m1}^L}}\partial{w_{m1}^L}&plus;\frac{\partial{z_m^L}}{\partial{w_{m2}^L}}\partial{w_{m2}^L}&plus;\frac{\partial{z_m^L}}{\partial{w_{m3}^L}}\partial{w_{m3}^L}&plus;\hdots&plus;\frac{\partial{z_m^L}}{\partial{w_{mp}^L}}\partial{w_{mp}^L})}" />
</p>

Like above, recall that our cost function can be written as followed
  
<p align="center">
  <img alt="cost function expanded" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}C=&\frac{1}{2}\Bigl[\left(y_1-\sigma\left(w_{11}^L\sigma(z_1^{L-1})&plus;w_{12}^L\sigma(z_2^{L-1})&plus;w_{13}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{1p}^L\sigma(z_p^{L-1})&plus;b_1^L\right)\right)^2\\&&plus;\left(y_2-\sigma\left(w_{21}^L\sigma(z_1^{L-1})&plus;w_{22}^L\sigma(z_2^{L-1})&plus;w_{23}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{2p}^L\sigma(z_p^{L-1})&plus;b_2^L\right)\right)^2\\&&plus;\left(y_3-\sigma\left(w_{31}^L\sigma(z_1^{L-1})&plus;w_{32}^L\sigma(z_2^{L-1})&plus;w_{33}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{3p}^L\sigma(z_p^{L-1})&plus;b_3^L\right)\right)^2&plus;\hdots\\&&plus;\left(y_m-\sigma\left(w_{m1}^L\sigma(z_1^{L-1})&plus;w_{m2}^L\sigma(z_2^{L-1})&plus;w_{m3}^L\sigma(z_3^{L-1})&plus;\hdots&plus;w_{mp}^L\sigma(z_p^{L-1})&plus;b_m^L\right)\right)^2\Bigr]\end{align*}}" />
</p>

where <img alt="p" src="https://latex.codecogs.com/svg.latex?p" /> is the number of <img alt="k" src="https://latex.codecogs.com/svg.latex?k" /> neurons in the <img alt="L-1" src="https://latex.codecogs.com/svg.latex?L-1" /> layer and <img alt="m" src="https://latex.codecogs.com/svg.latex?m" /> is the number of <img alt="j" src="https://latex.codecogs.com/svg.latex?j" /> neurons in the <img alt="L" src="https://latex.codecogs.com/svg.latex?L" />.

Finding the partial differentiation to this equation, we also get

<p align="center">
  <img alt="cost function partial with weight part 1" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\partial{C}=&\frac{\partial{C}}{\partial{w_{11}^L}}\partial{w_{11}^L}&plus;\frac{\partial{C}}{\partial{w_{12}^L}}\partial{w_{12}^L}&plus;\frac{\partial{C}}{\partial{w_{13}^L}}\partial{w_{13}^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{w_{1p}^L}}\partial{w_{1p}^L}\\&plus;&\frac{\partial{C}}{\partial{w_{21}^L}}\partial{w_{21}^L}&plus;\frac{\partial{C}}{\partial{w_{22}^L}}\partial{w_{22}^L}&plus;\frac{\partial{C}}{\partial{w_{23}^L}}\partial{w_{23}^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{w_{2p}^L}}\partial{w_{2p}^L}\\&plus;&\frac{\partial{C}}{\partial{w_{31}^L}}\partial{w_{31}^L}&plus;\frac{\partial{C}}{\partial{w_{32}^L}}\partial{w_{32}^L}&plus;\frac{\partial{C}}{\partial{w_{33}^L}}\partial{w_{33}^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{w_{3p}^L}}\partial{w_{3p}^L}&plus;\hdots\end{align*}}" />
</p>
<p align="center">
  <img alt="cost function partial with weight part 2" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{&plus;\frac{\partial{C}}{\partial{w_{m1}^L}}\partial{w_{m1}^L}&plus;\frac{\partial{C}}{\partial{w_{m2}^L}}\partial{w_{m2}^L}&plus;\frac{\partial{C}}{\partial{w_{m3}^L}}\partial{w_{m3}^L}&plus;\hdots&plus;\frac{\partial{C}}{\partial{w_{mp}^L}}\partial{w_{mp}^L}&plus;\text{other&space;terms}}" />
</p>

Equating, it become apparent that

<p align="center">
  <img alt="fourth fundamental equation" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\frac{\partial{C}}{\partial{w_{jk}^l}}&=\frac{\partial{z_j^l}}{\partial{w_{jk}^l}}\frac{\partial{C}}{\partial{z_j^l}}\\&=a_k^{l-1}\delta^l_j\end{align*}}" />
</p>

and like the past few fundamental equations, holds true for all layers!

## 5 - Backpropogation Using Gradient Descent

We use gradient descent to find the weights and biases which minimize the cost function. To do this, follow these steps:

1. Input a set of training examples

2. Set the input for the first layer, <img alt="a^x,1" src="https://latex.codecogs.com/svg.latex?a^{x,1}" />. For each training example, <img alt="x" src="https://latex.codecogs.com/svg.latex?x" />

  * Feedforwad: For each layer, <img alt="l equals 2,3,4,etc" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{l=2,&space;3,&space;4,\hdots,L}" /> find the values of the sigmoid outputs and arguments

<p align="center">
  <img alt="feedforward equations" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\mathbf{z}^{x,l}&=\mathbf{w}^l\mathbf{a}^{x,l-1}&plus;\mathbf{b}^l\\\mathbf{a}^{x,l}&=\sigma(\mathbf{z}^{x,l})\end{align*}}" />
</p>

  * Output error: For the last layer, find the error 

<p align="center">
  <img alt="error equations" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\delta}^{x,L}=\boldsymbol{\nabla}_aC_x\odot\boldsymbol{\sigma}'(z^{x,L})}" />
</p>

  * Backpropogate the error: For each layer, <img alt="l equals L-1,L-2,L-3,etc" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{l=L-1,&space;L-2,&space;L-3,\hdots,2}" /> find 

<p align="center">
  <img alt="error equations" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\boldsymbol{\delta}^{x,l}=((\mathbf{w}^{l&plus;1})^T\boldsymbol{\delta}^{x,l&plus;1})\odot\boldsymbol{\sigma}'(z^{x,l})}" />
</p>

3. Gradient descent: For each layer, <img alt="l equals L-1,L-2,L-3,etc" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{l=L-1,&space;L-2,&space;L-3,\hdots,2}" /> update the weights according to

<p align="center">
  <img alt="gradient descent weights" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\mathbf{w}^l\rightarrow\mathbf{w}^l-\frac{\eta}{n}\sum_{x=1}^n\boldsymbol{\delta}^{x,l}(\mathbf{a}^{x,l-1})^T\end{align*}}" />
</p>

and biases according to 

<p align="center">
  <img alt="gradient descent biases" src="https://latex.codecogs.com/svg.latex?\large\displaystyle{\begin{align*}\mathbf{b}^l\rightarrow\mathbf{b}^l-\frac{\eta}{n}\sum_{x=1}^n\boldsymbol{\delta}^{x,l}\end{align*}}" />
</p>

where <img alt="eta" src="https://latex.codecogs.com/svg.latex?\eta" /> is the user-defined learning rate. 

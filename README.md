# SimplePPL

SimplePPL is a simple probabilistic programming language designed to be
concise and easy to iterate models on. The backend is the Python package
[PyMC3](https://docs.pymc.io/) which provides the foundation and sampling algorithms.
Take a quick look at their documentation, especially if you are not familiar with
probablistic programming.

## Simple Example
The following simple program defines a simple Bayesian model
for the result of coin flips. This model defines a Beta prior (the conjugate prior for
the Bernoulli distribution) for the parameter `p`
of a Bernoulli random variable.
```
x = [0,1,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1]
p ~ Beta(1,1)
x ~ Bern(p)
```

Running this program returns statistics about the posterior distribution of the random variable `p`:
```
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (2 chains in 2 jobs)
NUTS: [p]
Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 12 seconds. [4000/4000 00:04<00:00 Sampling 2 chains, 0 divergences]
    mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
p  0.291  0.078   0.144    0.437      0.003    0.002     806.0    1356.0    1.0
```

## Installation

The two dependencies are [PyMC3](https://docs.pymc.io/) and [Lark](https://github.com/lark-parser/lark).
Lark is a Python parsing package and I used it for defining [the grammar](./grammar.lark) 
of the language.

## Structure of a SimplePPL Program

There are three types of statements in a SimplePPL Program. The first
type is a `data_assignment`, an example of which is the first line of code 
in the above example. A `data_assignment` assigns a data literal
to a variable and uses the `=` sign. The data literal is a multidimensional array, similar to Numpy arrays. SimplePPL program should begin by defining all the data using `data_assignment` statements
before defining the model.

The second type of statement is a `distributed_like` statement, which is used
to define the model. This statement includes a variable identifier followed
by a `~` and then one of the defined distributions (see below for full list).
The parameters of the distribution on the RHS can be literals or other random variables.

You can define the observed variables in your models by first having a `data_assignment` statement
for a variable, followed later by a `distributed_like` statement.

The last type of statement is an `assignment` which uses the token `:=`. 
The assignment statement is used for defining random variables which are
deterministic functions of other variables.


## Shapes

If not specified or inferred by data, all variables are scalars. To sample iid samples
from a distribution into a tensor of a given size, you can specify the shape
you want the variable to be in parentheses following the variable name. For instance the statement
```
x (3,2) ~ N(0,1)
```
defines a matrix `x` of random variables sampled iid from a standard Gaussian.
Shape arguments can also include shapes of other variables:
```
y (x#) ~ N(0,1)
```
will define a random variable `y` with the same shape of `x`. You can also specify 
components of the shapes of other variables. For instance,
```
z (2, x#0) ~ N(0,1)
```
defines a variable `z` that has shape `(2,3)`.

## Examples

See [this directory](./examples) for some examples of SimplePPL programs

## Supported distributions
* Bern - Bernoulli(p)
* N - Normal(mu,sigma)
* Unif - Continuoius uniform(lower, upper)
* Beta - Beta(alpha,beta)
* Pois - Poisson(mu)
* DUnif - Discrete Uniform(lower,upper)
* Gamma - Gamma(alpha, beta)
* Exp - Exponential(lambda)
* Geometric - Geometric(n)
* Binom - Binomial(n,p)

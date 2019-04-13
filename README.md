# neuralODE4j

Travis [![Build Status](https://travis-ci.org/DrChainsaw/AmpControl.svg?branch=master)](https://travis-ci.org/DrChainsaw/NeuralODE4j)
AppVeyor[![Build status](https://ci.appveyor.com/api/projects/status/wjdi11f4cmx32ir8?svg=true)](https://ci.appveyor.com/project/DrChainsaw/neuralode4j)

[![codebeat badge](https://codebeat.co/badges/d9e719b4-5465-4f08-9c14-f924691cdd86)](https://codebeat.co/projects/github-com-drchainsaw-neuralode4j-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d491774f94944895b6aa3e22b7aae8b3)](https://www.codacy.com/app/DrChainsaw/neuralODE4j?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DrChainsaw/neuralODE4j&amp;utm_campaign=Badge_Grade)
[![Maintainability](https://api.codeclimate.com/v1/badges/c0d216da01a0c8b8d615/maintainability)](https://codeclimate.com/github/DrChainsaw/neuralODE4j/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/c0d216da01a0c8b8d615/test_coverage)](https://codeclimate.com/github/DrChainsaw/neuralODE4j/test_coverage)

Implementation of neural Ordinary Differential Equations (ODE) built for [deeplearning4j](https://deeplearning4j.org/).

[[Arxiv](https://arxiv.org/abs/1806.07366)]

[[Pytorch repo by paper authors](https://github.com/rtqichen/torchdiffeq)]

[[Very good blog post](https://julialang.org/blog/2019/01/fluxdiffeq)]

## Getting Started

GIT clone and run with maven or in IDE.

```
git clone https://github.com/DrChainsaw/neuralODE4j
cd neuralODE4j
mvn install
```

I will try to create a maven artifact whenever I find the time for it. Please file an issue for this if you are interested.

Implementations of the MNIST and spiral generation toy experiments from the paper can be found under examples [[link]](./src/main/java/examples)

## Usage

The class [OdeVertex](./src/main/java/ode/vertex/conf/OdeVertex.java) is used to add an arbitrary graph of Layers or 
GraphVertices as an ODE block in a ComputationGraph.

OdeVertex extends GraphVertex and can be added to a GraphBuilder just as any other vertex. It has a similar API as 
GraphBuilder for adding layers and vertices. 

Example:
```
final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
        .graphBuilder()
        .addInputs("input")
        .setInputTypes(InputType.convolutional(9, 9, 3))
        .addLayer("normalLayer0",
                new Convolution2D.Builder(3, 3)
                        .nOut(32)
                        .convolutionMode(ConvolutionMode.Same).build(), "input")

        // Add an ODE block called "odeBlock" to the graph.
        .addVertex("odeBlock", 
                new OdeVertex.Builder(new NeuralNetConfiguration.Builder(), "odeLayer0", new BatchNormalization.Builder().build())
                
                // OdeVertex has a similar API as GraphBuilder for adding new layers/vertices to the OdeBlock
                .addLayer("odeLayer1", new Convolution2D.Builder(3, 3)
                        .nOut(32)
                        .convolutionMode(ConvolutionMode.Same).build(), "odeLayer0")
                
                // Add more layers and vertices as desired
                
                // Build the OdeVertex. The resulting "inner graph" will be treated as an ODE
                .build(), "normalLayer0")

        // Layers/vertices can be added to the graph after the ODE block
        .addLayer("normalLayer1", new BatchNormalization.Builder().build(), "odeBlock")
        .setOutputs("output")
        .addLayer("output", new CnnLossLayer(), "normalLayer1")
        .build());
```

An inherent constraint to the method itself is that the output of the last layer in the OdeVertex must have the exact 
same shape as the input to the first layer in the OdeVertex. 

Note that OdeVertex.Builder requires a NeuralNetConfiguration.Builder as constructor input. This is because DL4J does 
not set graph-wise default values for things like updaters and weight initialization for non.layer vertices so the only
way to apply them to the Layers of the OdeVertex is to pass in the global configuration. Putting it as a required 
constructor argument will hopefully make this harder to forget. It is of course possible to have a separate set of 
default values for the layers of the OdeVertex by just giving it another NeuralNetConfiguration.Builder. 

Method for solving the ODE can be configured:

```
new OdeVertex.Builder(...)
    .odeConf(new FixedStep(
                      new DormandPrince54Solver(),
                      Nd4j.arange(0,2))) // Integrate between t = 0 and t = 1
```

Currently, the only ODE solver implementation which is integrated with Nd4j is [DormandPrince54Solver](./src/main/java/ode/solve/impl/DormandPrince54Solver.java), 
It is however possible to use FirstOrderIntegrators from apache.commons:commons-math3 through [FirstOrderSolverAdapter](./src/main/java/ode/solve/commons/FirstOrderSolverAdapter.java)
at the cost of slower training and inference speed.

Time steps to solve the ODE for can also be input from another vertex in the graph:
```
new OdeVertex.Builder(...)
    .odeConf(new InputStep(solverConf, 1)) // Number "1" refers to input "time" on the line below
    .build(), "someLayer", "time");  
```

Note that time must be a vector meaning it can not be minibatched; It has to be the same for all examples in a minibatch. This is because the implementation uses the minibatching approach from 
section 6 in the paper where all examples in the batch are concatenated into one state. If one time sequence per example is desired this 
can be achieved by using minibatch size of 1.  

Gradients for loss with respect to time will be output from the vertex when using time as input but will be set to 0 by default to save computation. To have them computed, set needTimeGradient to true:

```
final boolean needTimeGradient = true;
new OdeVertex.Builder(...)
    .odeConf(new InputStep(solverConf, 1, true, needTimeGradient))
    .build(), "someLayer", "time");  
```

I have not seen these being used for anything in the original implementation and if used, some extra measure is most likely required to ensure that time is always strictly increasing or decreasing.

In either case, the minimum number of elements in the time vector is two. If more than two elements are given the output of the OdeVertex 
will have one more dimension compared to the input (corresponding to each time element). 

For example, if the graph in the OdeVertex is the function `f = dz/dt` and `time` is the sequence `t0, t1, ..., tN-1` 
with `N > 2` then the output of the OdeVertex will be (an approximation of) the sequence `z(t0), z(t1), ... , z(tN-1)`. 
Note that `z(t0)` is also the input to the OdeVertex.
 
The exact mapping to dimensions depends on the shape of the input. Currently the following mappings are supported:

| Input shape               | Output shape                  |
|---------------------------|-------------------------------| 
| `B x H (dense/FF)`        | `B x H x t  (RNN)`            |
| `B x H x T(RNN)`          | `Not supported`               |
| `B x D x H x W (conv 2D) `| `B x D x H x W x t  (conv 3D)`|

The current time step of the ODE solver can be used as an input to layers in the OdeVertex:

```
new OdeVertex.Builder(...)
    .addTimeLayer("someName", someLayer)
```
Since the current time is a scalar, a [DuplicateScalarToShape](src/main/java/util/preproc/DuplicateScalarToShape.java) 
preprocessor is automatically added when doing this.

For vertices the following method is used:
```
new OdeVertex.Builder(...)
    .addTimeVertex(someName, someGraphVertex, inputs)
```
Time will be added as the last input to the vertex. A [ShapeMatchVertex](src/main/java/ode/vertex/conf/ShapeMatchVertex.java)
can be used to adapt the shape of the time input for vertices which don't support broadcasting.

### Limitations

The implementation does not use Nd4js SameDiff and therefore automatic differentiation is not possible. As a consequence, 
back propagation though the OdeVertex is only possible using the adjoint method. 

For the same reason, continuous normalizing flows is not supported as this requires back-propagation through the 
derivative of a forward pass though the OdeVertex (as far as I can understand).

Reason for not using SameDiff is that it does not have GPU support in Nd4j 1.0.0-beta3. 

### Prerequisites

Maven and GIT. Project uses ND4Js CUDA 10 backend as default which requires [CUDA 10](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn).
To use CPU backend instead, set the maven property backend-CPU:

```
mvn install -P backend-CPU
```

## Contributing

All contributions are welcome. Head over to the issues page and either add a new issue or pick up and existing one.

## Versioning

TBD.

## Authors

* **Christian Skärby** - *Initial work* - [DrChainsaw](https://github.com/DrChainsaw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud for a very cool and inspiring paper
* Deeplearning4j for neural nets

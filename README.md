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

The class OdeVertex is used to add an arbitrary graph of Layers or GraphVertices as an ODE block in a ComputationGraph. OdeVertex
extends GraphVertex and can be added to a GraphBuilder just as any other vertex. It has a similar API as GraphBuilder for adding 
layers and vertices. 

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
                        new OdeVertex.Builder("odeLayer0", new BatchNormalization.Builder().build())
                        
                        // OdeVertex has a similar API as GraphBuilder for adding new layers/vertices to the OdeBlock
                        .addLayer("odeLayer1", new Convolution2D.Builder(3, 3)
                                .nOut(32)
                                .convolutionMode(ConvolutionMode.Same).build(), "odeLayer0")
                        
                        // Add more layers and vertices as desired
                        
                        // Build the OdeVertex. All layers added to it will be treated as an ODE
                        .build(), "normalLayer0")

                // Layers can be added to the graph after the ODE block
                .addLayer("normalLayer1", new BatchNormalization.Builder().build(), "odeBlock")
                .setOutputs("output")
                .addLayer("output", new CnnLossLayer(), "normalLayer1")
                .build());
```

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

Time can also be input from another vertex:
```
new OdeVertex.Builder(...)
    .odeConf(new InputStep(solverConf, 1)) // Number "1" refers to input "time" on the line below
    .build(), "someLayer", "time");  
```

Implementations of the MNIST and spiral generation toy experiments from the paper can be found under examples [[link]](./src/main/java/examples)

### Prerequisites

Maven and GIT. Project uses ND4Js CUDA 10 backend as default which requires [CUDA 10](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn).
To use CPU backend instead, set the maven property backend-CPU:

```
mvn install -P backend-CPU
```

## Contributing

All contributions are welcome. Head over to the issues page and either add a new issue or pick up and existing one.

## Versioning

TBD. I will try to add a maven artifact whenever I find the time for it.

## Authors

* **Christian Skärby** - *Initial work* - [DrChainsaw](https://github.com/DrChainsaw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud for a very cool and inspiring paper
* Deeplearning4j for neural nets

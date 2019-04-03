# Examples

## MNIST

Reimplementation of the MNIST experiment from the [original repo](https://github.com/rtqichen/torchdiffeq/tree/master/examples).

To run the ResNet reference model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.mnist.Main" -Dexec.args="resnet"
```
To run the ODE net model, use the following command:

```
mvn exec:java -Dexec.mainClass="examples.mnist.Main" -Dexec.args="odenet"
```

Running from the IDE is also possible in which case resnet/odenet must be set as program arguments.

Use -help for full list of command line arguments:

```
mvn exec:java -Dexec.mainClass="examples.mnist.Main" -Dexec.args="-help"
```

Performance (approx):

| Model  | Error |
| ------ | ----- |
| resnet | 0.5%  |
| odenet | 0.5%  |
| stem   | 0.5%  |

Model "stem" is using the resnet option with zero resblocks after the downsampling layers. This indicates that neither the residual blocks nor the ode block seems to be contributing much to the performance in this simple experiment. Performance also varies about +-0.1% for each run of the same model.

## Spiral demo

Reimplementation of the spiral generation experiment from the [original repo](https://github.com/rtqichen/torchdiffeq/tree/master/examples).

To run the ODE net model, use the following command:

```
mvn exec:java -Dexec.mainClass="examples.spiral.Main" -Dexec.args="odenet"
```

Running from the IDE is also possible in which case odenet must be set as program arguments.

Use -help for full list of command line arguments:

```
mvn exec:java -Dexec.mainClass="examples.spiral.Main" -Dexec.args="-help"
```

Note that this example tends to run faster on CPU than on GPU, probably due to the relatively low number of parameters. Example:

```
mvn -P backend-CPU exec:java -Dexec.mainClass="examples.spiral.Main" -Dexec.args="odenet"
```

Furthermore, original implementation does not use the adjoint method for back propagation in this example and instead does backpropagation through the operations of the ODE solver. Backpropagation through the ODE solver is not supported in this project as of yet. 

## CIFAR 10

Image classification experiment comparing the [Inception ResNet V1](https://arxiv.org/pdf/1602.07261.pdf) architecture with the same architecture but with the residual blocks replaced by ODE vertices. The stem part of the network is not used as the CIFAR 10 pictures are only 32x32 to begin with so the input is fed directly into the first type A block.

As the original architecture has three different types of blocks (A, B and C) and reduces the size of the input maps between each block. To stay analogous to this, the ODE network uses three ODE vertices, one wrapping each block type. The is also an additional batch normalization added just after the input ReLU and after each ODE/add vertex as I could not get convergence in either architecture otherwise. 

To run the ResNet reference model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.cifar10.Main" -Dexec.args="resnet"
```

To run the ODE model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.cifar10.Main" -Dexec.args="odenet -trainTime -useABlock -useBBlock -useCBlock"
```

It is possible to run without any of the A,B,C blocks by omitting the respective flag.

Use -help for full list of command line arguments:

```
mvn exec:java -Dexec.mainClass="examples.cifar10.Main" -Dexec.args="-help"
```

Performance:

| Model  | Error  |
| ------ | ------ |
| resnet | TBD  |
| odenet | TBD    |

Training time and memory (using a GTX 980 ti):

| Model  | Time   | Memory used for training |
| ------ | ------ | -------|
| resnet | TBD    | 5.3 GB |
| odenet | TBD    | 3.5 GB |

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

## Augmented neural ODEs

Implementation of toy experiments from the [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681) paper. 

To run the ResNet reference model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.anode.Main" -Dexec.args="-2D resnet"
```
To run the ODE net model, use the following command:

```
mvn exec:java -Dexec.mainClass="examples.anode.Main" -Dexec.args="-2D odenet"
```

To run the augmented ODE net model, use the following command:

```
mvn exec:java -Dexec.mainClass="examples.anode.Main" -Dexec.args="-2D odenet -nrofAugmentDims 5"
```

Running from the IDE is also possible in which case resnet/odenet must be set as program arguments.

Use -help for full list of command line arguments:

```
mvn exec:java -Dexec.mainClass="examples.anode.Main" -Dexec.args="-help"
```


## CIFAR 10

Image classification experiment comparing the [Inception ResNet V1](https://arxiv.org/pdf/1602.07261.pdf) architecture 
with the same architecture when the residual blocks replaced by ODE vertices. The stem part of the network is not 
used as the CIFAR 10 pictures are only 32x32 to begin with so the input is fed directly into the first type A block.

As the original architecture has three different types of blocks (A, B and C) and reduces the size of the input maps 
between each block. To stay analogous to this, the ODE network uses three ODE vertices, one wrapping each block type.

As a second reference model, a small inception net with just one of each block was also tested. This model has a similar
number of parameters as the ODE model.

The standard data augmentation of applying a uniform [0,4] shift and 50% propability for horizontal flip are applied to 
the training set. I forgot to standardize the training set, but given the abundance of batch normalization it might not 
have mattered much.  

To run the full Inception ResNet reference model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.cifar10.Main" -Dexec.args="resnet"
```

To run the small Inception ResNet reference model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.cifar10.Main" -Dexec.args="resnet -nrofABlocks 1 -nrofBBlocks 1 -nrofCBlocks 1"
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

Performance after 50 epochs:

| Model        | Test error   |
| ------------ | ------------ |
| full resnet  | 11.3%        |
| small resnet | 23.8%        |
| odenet       | 15.9%        |

Training time and memory (using a GTX 980 ti):

| Model         | Time   | Memory used for training |
| ------------- | ------ | -------------------------|
| full resnet   | 20h    | 5.3 GB                   |
| small resnet  | 10h    | 3.5 GB                   |
| odenet        | 240h(!)| 4.1 GB                   |

As shown above, odenet takes about 10 times longer to train compared to the baseline and does not achieve quite the same
performance. The latter can perhaps be explained to some extent by the hyper parameters for the baseline being tuned to 
that architecture. It is however significantly better than the small resnet.

It should also be noted that the test error for all models is a lot higher than 
[reported SOTA](https://en.wikipedia.org/wiki/CIFAR-10#Research_Papers_Claiming_State-of-the-Art_Results_on_CIFAR-10). 
This could be due to longer training and use of ensembles. 
As an example, the [wide resnet paper](https://arxiv.org/abs/1605.07146) results (4% error)  are after 200 epochs. 
Figure 2 of the paper show similar test error as the table above after 50 epochs. Similar goes for the
[densenet](https://arxiv.org/abs/1608.06993) paper.
 
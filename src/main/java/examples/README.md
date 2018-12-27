# Examples

# MNIST

Reimplementation of the MNIST experiment from the [original repo] (https://github.com/rtqichen/torchdiffeq/tree/master/examples).

To run the ResNet reference model, use the following command:
```
mvn exec:java -Dexec.mainClass="examples.mnist.Main" -Dexec.args="resnet"
```
To run the ODE net model, use the following command:

```
mvn exec:java -Dexec.mainClass="examples.mnist.Main" -Dexec.args="odenet"
```

Running from the IDE is also possible in which case resnet/odenet must be set as program arguments.


Performance: TBD
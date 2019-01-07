# Examples

# MNIST

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


Performance (approx):

| Model  | Error |
| ------ | ----- |
| resnet | 0.5%  |
| odenet | 0.5%  |
| stem   | 0.5%  |

Model "stem" is using the resnet option with zero resblocks after the downsampling layers. This indicates that neither the residual blocks nor the ode block seems to be contributing much to the performance in this simple experiment. Performance also varies about +-0.1% for each run of the same model.

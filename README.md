# neuralODE4j

Implementation of neural ordinary differential equations built for deeplearning4j.

[[Arxiv](https://arxiv.org/abs/1806.07366)]

[[Pytorch repo by paper authors](https://github.com/rtqichen/torchdiffeq)]

NOTE: This is very much a work in progress and given that I haven't touched a differential equation since school chances
are that there are conceptual misunderstandings.

There is currently a critical [memory issue](https://github.com/DrChainsaw/neuralODE4j/issues/5) which prevents validation.
As such, the implementation is basically unverified except for indications that test score steadily decreases in the MNIST example.

## Getting Started

GIT clone and run with maven or in IDE.

```
git clone https://github.com/DrChainsaw/neuralODE4j
cd neuralODE4j
mvn install
```

Currently only the MNIST toy experiment from the paper is implemented [[link]](./examples)

### Prerequisites

Maven and GIT. Project uses ND4Js CUDA 10 backend as default which requires [CUDA 10](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn).
To use CPU backend instead, set the maven property backend-CPU (e.g. through the -P flag when running from command line).

## Contributing

All contributions are welcome. Head over to the issues page and either add a new issue or pick up and existing one.

## Versioning

TBD

## Authors

* **Christian Skärby** - *Initial work* - [DrChainsaw](https://github.com/DrChainsaw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud for a very cool and inspiring paper
* Deeplearning4j for neural nets

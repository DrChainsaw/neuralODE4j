package examples.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Standard convolution stem block from https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 */
public class ConvStem implements Block {

    private final long nrofKernels;

    public ConvStem(long nrofKernels) {
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {
        builder
                .addInputs("input")
                .inputPreProcessor("firstConv", new FeedForwardToCnnPreProcessor(28, 28))
                .addLayer("firstConv",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .convolutionMode(ConvolutionMode.Same)
                                .activation(new ActivationIdentity())
                                .build(), "input")
                .addLayer("firstNorm",
                        new BatchNormalization.Builder()
                                .activation(new ActivationReLU()).build(), "firstConv")
                .addLayer("secondConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2, 2)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "firstNorm")
                .addLayer("secondNorm",
                        new BatchNormalization.Builder()
                                .activation(new ActivationReLU()).build(), "secondConv")
                .addLayer("thirdConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2, 2)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "secondNorm");

        return "thirdConv";
    }


}

package examples.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Standard convolution stem block from https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class ConvStem implements Block {

    private final long nrofKernels;

    public ConvStem(long nrofKernels) {
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {
        builder
                .addLayer("firstConv",
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .build(), prev)
                .addLayer("firstNorm",
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build(), "firstConv")
                .addLayer("secondConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2, 2)
                                .nOut(nrofKernels)
                                .padding(1,1)
                                // Reference implementation uses identity, but performance becomes very bad if used here (?!)
                                .activation(new ActivationReLU())
                                //.activation(new ActivationIdentity())
                                .build(), "firstNorm")
                .addLayer("secondNorm",
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU())
                                .build(), "secondConv")
                .addLayer("thirdConv",
                        new Convolution2D.Builder(4, 4)
                                .stride(2, 2)
                                .nOut(nrofKernels)
                                .padding(1,1)
                                .activation(new ActivationIdentity())
                                .build(), "secondNorm");

        return "thirdConv";
    }
}

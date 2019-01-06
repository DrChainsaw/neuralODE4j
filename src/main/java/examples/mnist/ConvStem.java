package examples.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.nd4j.linalg.activations.impl.ActivationReLU;

import static examples.mnist.LayerUtil.*;

/**
 * Standard convolution stem block from https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class ConvStem implements Block {

    private final int nrofKernels;

    public ConvStem(int nrofKernels) {
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {
        builder
                .addLayer("firstConv",
                        conv3x3(nrofKernels), prev)
                .addLayer("firstNorm",
                        norm(nrofKernels), "firstConv")
                .addLayer("secondConv",
                        // Reference implementation uses identity, but performance becomes very bad if used here (?!)
                        conv4x4DownSample(nrofKernels, new ActivationReLU()), "firstNorm")
                .addLayer("secondNorm",
                        norm(nrofKernels), "secondConv")
                .addLayer("thirdConv",
                        conv4x4DownSample(nrofKernels), "secondNorm");

        return "thirdConv";
    }
}

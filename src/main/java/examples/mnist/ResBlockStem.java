package examples.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;

/**
 * Residual convolution stem block from https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
public class ResBlockStem implements Block {

    private final int nrofKernels;

    public ResBlockStem(int nrofKernels) {
        this.nrofKernels = nrofKernels;
    }

    @Override
    public String add(String prev, ComputationGraphConfiguration.GraphBuilder builder) {

        String next = "inputConv";
        builder.addLayer(next, new Convolution2D.Builder(3, 3)
                .nOut(nrofKernels)
                .activation(new ActivationIdentity())
                .build(), prev);

        next = addResBlock(builder, next, 0);
        next = addResBlock(builder, next, 1);

        return next;
    }


    private String addResBlock(ComputationGraphConfiguration.GraphBuilder builder, String prev, int cnt) {
        builder
                .addLayer("firstNormStem_" + cnt,
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build(), prev)
                .addLayer("downSample_" + cnt,
                        new Convolution2D.Builder(1, 1)
                                .nOut(nrofKernels)
                                .stride(2,2) // 1x1 conv with stride 2? Won't this just remove 50% of information?
                                .hasBias(false)
                                .activation(new ActivationIdentity())
                                .build() , "firstNormStem_" + cnt)
                .addLayer("firstConvStem_" + cnt,
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .hasBias(false)
                                .stride(2,2)
                                // Reference implementation uses identity, but performance becomes very bad if used here (?!)
                                .activation(new ActivationReLU())
                                //.activation(new ActivationIdentity())
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "firstNormStem_" + cnt)
                .addLayer("secondNormStem_" + cnt,
                        new BatchNormalization.Builder()
                                .nOut(nrofKernels)
                                .activation(new ActivationReLU()).build(), "firstConvStem_" + cnt)
                .addLayer("secondConvStem_" + cnt,
                        new Convolution2D.Builder(3, 3)
                                .nOut(nrofKernels)
                                .activation(new ActivationIdentity())
                                .hasBias(false)
                                .convolutionMode(ConvolutionMode.Same)
                                .build(), "secondNormStem_" + cnt)
                .addVertex("stemAdd_" + cnt, new ElementWiseVertex(ElementWiseVertex.Op.Add), "downSample_" + cnt, "secondConvStem_" + cnt);
        return "stemAdd_" + cnt;
    }
}

package examples.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;

import static examples.mnist.LayerUtil.*;

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
        builder.addLayer(next, conv3x3(nrofKernels), prev);

        next = addResBlock(builder, next, 0);
        next = addResBlock(builder, next, 1);

        return next;
    }


    private String addResBlock(ComputationGraphConfiguration.GraphBuilder builder, String prev, int cnt) {
        builder
                .addLayer("firstNormStem_" + cnt,
                        norm(nrofKernels), prev)
                .addLayer("downSample_" + cnt,
                        conv1x1DownSample(nrofKernels), "firstNormStem_" + cnt)
                .addLayer("firstConvStem_" + cnt,
                        conv3x3DownSample(nrofKernels), "firstNormStem_" + cnt)
                .addLayer("secondNormStem_" + cnt,
                        norm(nrofKernels), "firstConvStem_" + cnt)
                .addLayer("secondConvStem_" + cnt,
                        conv3x3Same(nrofKernels), "secondNormStem_" + cnt)
                .addVertex("stemAdd_" + cnt, new ElementWiseVertex(ElementWiseVertex.Op.Add), "downSample_" + cnt, "secondConvStem_" + cnt);
        return "stemAdd_" + cnt;
    }
}

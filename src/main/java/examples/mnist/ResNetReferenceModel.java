package examples.mnist;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static examples.mnist.LayerUtil.*;

/**
 * ResNet reference model for comparison. Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
 *
 * @author Christian Skarby
 */
@Parameters(commandDescription = "Configuration for image classification using a number of residual blocks")
public class ResNetReferenceModel implements ModelFactory {

    private static final Logger log = LoggerFactory.getLogger(ResNetReferenceModel.class);

    @ParametersDelegate
    private StemSelection stemSelection = new StemSelection();

    @Parameter(names = "-nrofResBlocks", description = "Number of residual blocks to use")
    private int nrofResBlocks = 6;

    @Parameter(names = "-nrofKernels", description = "Number of filter kernels in each convolution layer")
    private int nrofKernels = 64;

    private final GraphBuilder builder;

    public ResNetReferenceModel() {
        this.builder = initGraphBuilder(666);
    }

    @Override
    public ComputationGraph create() {

        log.info("Create model");

        String next = stemSelection.get(nrofKernels).add(builder.getNetworkInputs().get(0), builder);
        for (int i = 0; i < nrofResBlocks; i++) {
            next = addResBlock(next, i);
        }
        new Output(nrofKernels).add(next, builder);
        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        return graph;
    }

    @Override
    public String name() {
        return "resnet_" + nrofResBlocks + "_" + stemSelection.name();
    }


    private String addResBlock(String prev, int cnt) {
        builder
                .addLayer("normFirst_" + cnt,
                        norm(64), prev)
                .addLayer("convFirst_" + cnt,
                        conv3x3Same(nrofKernels), "normFirst_" + cnt)
                .addLayer("normSecond_" + cnt,
                        norm(64), "convFirst_" + cnt)
                .addLayer("convSecond_" + cnt,
                        conv3x3Same(nrofKernels), "normSecond_" + cnt)
                .addVertex("add_" + cnt, new ElementWiseVertex(ElementWiseVertex.Op.Add), prev, "convSecond_" + cnt);
        return "add_" + cnt;
    }
}

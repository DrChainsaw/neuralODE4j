package examples.anode;

import com.beust.jcommander.Parameter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Create the ResNet model described in section E1.1 and E2.1 in https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
public class ResNetModelFactory implements ModelFactory {

    @Parameter(names = "-nrofResBlock", description = "Number of residual blocks to stack")
    private long nrofResBlocks = 5;

    @Parameter(names = "-nrofHidden", description = "Number of hidden units")
    private long nrofHidden = 32;

    @Override
    public Model create(long nrofInputDims) {
        String next = "input";
        final ComputationGraphConfiguration.GraphBuilder builder = LayerUtil.initGraphBuilder(nrofInputDims)
                .addInputs(next);

        final List<String> resBlockNames = new ArrayList<>();
        for(int i = 0; i < nrofResBlocks; i++) {
            final String name = "resblock_" + i;
            next = new ResBlock(name, new MlpBlock(name + ".", nrofHidden, nrofInputDims))
                    .add(new GraphBuilderWrapper.Wrap(builder), next);
            resBlockNames.add(next);
        }

        final String output = new LossBlock().add(new GraphBuilderWrapper.Wrap(builder), next);
        builder.setOutputs(output);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();

        return new ResNetModel(graph, resBlockNames);
    }
}

package examples.anode;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;

import java.util.Arrays;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MlpBlock}
 *
 * @author Christian Skarby
 */
public class MlpBlockTest {

    /**
     * Test that layers are added in the correct order and have the right input and output sizes
     */
    @Test
    public void add() {
        final long nrofOutputs = 3;
        final long nrofHidden = 8;
        final ComputationGraphConfiguration.GraphBuilder graphBuilder = LayerUtil.initGraphBuilder(nrofOutputs)
                .addInputs("input")
                .allowNoOutput(true);

        final String out = new MlpBlock(nrofHidden, nrofOutputs).add(new GraphBuilderWrapper.Wrap(graphBuilder), "input");
        assertEquals("Incorrect final layer!", "out", out);

        final ComputationGraph graph = new ComputationGraph(graphBuilder.build());
        graph.init();

        assertEquals("Incorrect order!", Arrays.asList("input", "h1", "h2", "out"), graph.getConfiguration().getTopologicalOrderStr());

        assertEquals("Incorrect input size!", nrofOutputs, graph.layerInputSize("h1"));
        assertEquals("Incorrect output size!", nrofHidden, graph.layerSize("h1"));

        assertEquals("Incorrect input size!", nrofHidden, graph.layerInputSize("h2"));
        assertEquals("Incorrect output size!", nrofHidden, graph.layerSize("h2"));

        assertEquals("Incorrect input size!", nrofHidden, graph.layerInputSize("out"));
        assertEquals("Incorrect output size!", nrofOutputs, graph.layerSize("out"));
    }
}
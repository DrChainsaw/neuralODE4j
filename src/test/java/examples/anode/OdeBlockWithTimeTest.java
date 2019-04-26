package examples.anode;

import ode.solve.conf.DummyIteration;
import ode.vertex.conf.helper.FixedStep;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link OdeBlockWithTime}
 *
 * @author Christian Skarby
 */
public class OdeBlockWithTimeTest {

    /**
     * Test that an ODE vertex is added correctly
     */
    @Test
    public void add() {
        final long nrofOutputs = 5;
        final ComputationGraphConfiguration.GraphBuilder graphBuilder = LayerUtil.initGraphBuilder(nrofOutputs)
                .addInputs("input");
        final String out = new OdeBlockWithTime(
                graphBuilder.getGlobalConfiguration(),
                new DummyBlock(nrofOutputs),
                new FixedStep(new DummyIteration(2), Nd4j.arange(2).reshape(1, 2), true))
                .add(new GraphBuilderWrapper.Wrap(graphBuilder), "input");

        assertEquals("Incorrect outname!", "ode", out);

        graphBuilder
                .setOutputs("output")
        .addLayer("output", new LossLayer.Builder().activation(new ActivationIdentity()).build(), out);
        final ComputationGraph graph = new ComputationGraph(graphBuilder.build());
        graph.init();

        assertEquals("Incorrect order!", Arrays.asList("input", "ode", "output"), graph.getConfiguration().getTopologicalOrderStr());

        assertEquals("Incorrect number of outputs!", nrofOutputs, graph.outputSingle(Nd4j.arange(nrofOutputs)).length());
    }

    private static class DummyBlock implements Block {

        private final long nrofOutputs;

        private DummyBlock(long nrofOutputs) {
            this.nrofOutputs = nrofOutputs;
        }

        @Override
        public String add(GraphBuilderWrapper builder, String... prev) {
            builder.addLayer("dummy", new DenseLayer.Builder()
                    .nOut(nrofOutputs)
                    .build(), prev);
            return "dummy";
        }
    }
}
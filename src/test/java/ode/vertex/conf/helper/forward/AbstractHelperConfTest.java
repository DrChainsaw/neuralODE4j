package ode.vertex.conf.helper.forward;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;

abstract class AbstractHelperConfTest {

    /**
     * Create a {@link OdeHelperForward} which shall be tested
     *
     * @return a new {@link OdeHelperForward}
     */
    abstract OdeHelperForward create();

    /**
     * Create inputs from a given input array
     *
     * @param input input array
     * @return all needed inputs
     */
    abstract INDArray[] createInputs(INDArray input);

    /**
     * Test serialization and deserialization
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final OdeHelperForward conf = create();
        final String json = NeuralNetConfiguration.mapper().writeValueAsString(conf);
        final OdeHelperForward newConf = NeuralNetConfiguration.mapper().readValue(json, OdeHelperForward.class);
        assertEquals("Did not deserialize into the same thing!", conf, newConf);
    }

    @Test
    public void instantiateAndSolveConv() {
        final ode.vertex.impl.helper.forward.OdeHelperForward helper = create().instantiate();
        final INDArray output = helper.solve(createGraph(), LayerWorkspaceMgr.noWorkspaces(), createInputs(Nd4j.randn(new long[] {5, 2, 3, 3})));
        assertNotEquals("Expected non-zero output!", 0, output.sumNumber().doubleValue() ,1e-10);
    }

    private ComputationGraph createGraph() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .allowNoOutput(true)
                .addInputs("input")
                .setInputTypes(InputType.convolutional(9, 9, 2))
                .addLayer("1", new Convolution2D.Builder(3, 3).nOut(2).convolutionMode(ConvolutionMode.Same).build(), "input")
                .build());
        graph.init();
        return graph;
    }
}

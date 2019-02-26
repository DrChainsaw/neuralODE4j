package ode.vertex.conf.helper.backward;

import ode.vertex.impl.NonContiguous1DView;
import ode.vertex.impl.helper.backward.OdeHelperBackward.InputArrays;
import ode.vertex.impl.helper.backward.OdeHelperBackward.MiscPar;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotEquals;

abstract class AbstractHelperConfTest {

    private final static InputType convInput = InputType.convolutional(9,9,2);

    /**
     * Create a {@link OdeHelperBackward} which shall be tested
     *
     * @param nrofTimeSteps Number of time steps to create helper for
     * @return a new {@link OdeHelperBackward}
     */
    abstract OdeHelperBackward create(int nrofTimeSteps);

    /**
     * Create inputs from a given input array
     *
     * @param input input array
     * @param nrofTimeSteps number of time steps to create input for
     * @return all needed inputs
     */
    abstract INDArray[] createInputs(INDArray input, int nrofTimeSteps);

    /**
     * Test serialization and deserialization
     */
    @Test
    public void serializeDeserializeSingleStep() throws IOException {
        final OdeHelperBackward conf = create(2);
        final String json = NeuralNetConfiguration.mapper().writeValueAsString(conf);
        final OdeHelperBackward newConf = NeuralNetConfiguration.mapper().readValue(json, OdeHelperBackward.class);
        assertEquals("Did not deserialize into the same thing!", conf, newConf);
    }

    /**
     * Test serialization and deserialization
     */
    @Test
    public void serializeDeserializeMultiStep() throws IOException {
        final OdeHelperBackward conf = create(5);
        final String json = NeuralNetConfiguration.mapper().writeValueAsString(conf);
        final OdeHelperBackward newConf = NeuralNetConfiguration.mapper().readValue(json, OdeHelperBackward.class);
        assertEquals("Did not deserialize into the same thing!", conf, newConf);
    }

    /**
     * Test that helper can be instantiated and that it does something
     */
    @Test
    public void instantiateAndSolveConvSingleStep() {
        final int nrofTimeSteps = 2;
        final ode.vertex.impl.helper.backward.OdeHelperBackward helper = create(nrofTimeSteps).instantiate();
        final ComputationGraph graph = createGraph();
        final InputArrays input = getTestInputArrays(nrofTimeSteps, graph);
        final Pair<Gradient, INDArray[]> output = helper.solve(graph, input, new MiscPar(
                false,
                LayerWorkspaceMgr.noWorkspaces(),
                "grad"));
        assertNotEquals("Expected non-zero param gradient!", 0, output.getFirst().gradient().sumNumber().doubleValue() ,1e-10);
        for(INDArray inputGrad: output.getSecond()) {
            assertNotEquals("Expected non-zero param gradient!", 0, inputGrad.sumNumber().doubleValue() ,1e-10);
        }
    }

    /**
     * Test that helper can be instantiated and that it does something
     */
    @Test
    public void instantiateAndSolveConvMultiStep() {
        final int nrofTimeSteps = 7;
        final ode.vertex.impl.helper.backward.OdeHelperBackward helper = create(nrofTimeSteps).instantiate();
        final ComputationGraph graph = createGraph();
        final InputArrays input = getTestInputArrays(nrofTimeSteps, graph);
        final Pair<Gradient, INDArray[]> output = helper.solve(graph, input, new MiscPar(
                false,
                LayerWorkspaceMgr.noWorkspaces(),
                "grad"));
        Nd4j.getExecutioner().commit();
        assertNotEquals("Expected non-zero param gradient!", 0, output.getFirst().gradient().sumNumber().doubleValue() ,1e-10);
        for(INDArray inputGrad: output.getSecond()) {
            assertNotEquals("Expected non-zero param gradient!", 0, inputGrad.sumNumber().doubleValue() ,1e-10);
        }
    }

    private ComputationGraph createGraph() {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .allowNoOutput(true)
                .addInputs("input")
                .setInputTypes(convInput)
                .addLayer("1", new Convolution2D.Builder(3, 3).nOut(2).convolutionMode(ConvolutionMode.Same).build(), "input")
                .build());
        graph.init();
        graph.initGradientsView();
        return graph;
    }

    @NotNull
    private InputArrays getTestInputArrays(int nrofTimeSteps, ComputationGraph graph) {

        final int batchSize = 3;
        final long[] shape = convInput.getShape(true);
        shape[0] = batchSize;

        final int nrofTimeStepsToUse = nrofTimeSteps == 2 ? 1 : nrofTimeSteps;
        final long[] outputShape = nrofTimeSteps == 2 ? shape : new long[]{ batchSize, nrofTimeStepsToUse, shape[1], shape[2], shape[3]};


        final INDArray input = Nd4j.arange(batchSize * convInput.arrayElementsPerExample()).reshape(shape);
        final INDArray output = Nd4j.arange(batchSize * nrofTimeStepsToUse * convInput.arrayElementsPerExample())
                .reshape(outputShape);
        final INDArray epsilon = Nd4j.ones(outputShape);
        final NonContiguous1DView realGrads = new NonContiguous1DView();
        realGrads.addView(graph.getGradientsViewArray());

        return new InputArrays(
                createInputs(input, nrofTimeSteps),
                output,
                epsilon,
                realGrads
        );
    }
}

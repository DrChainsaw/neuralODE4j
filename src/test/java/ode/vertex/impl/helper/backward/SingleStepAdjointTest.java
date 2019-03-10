package ode.vertex.impl.helper.backward;

import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import ode.vertex.impl.gradview.NonContiguous1DView;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link SingleStepAdjoint}
 *
 * @author Christian Skarby
 */
public class SingleStepAdjointTest {

    /**
     * Smoke test for backwards solve without time gradients
     */
    @Test
    public void solveNoTime() {
        final int nrofInputs = 7;
        final ComputationGraph graph = getTestGraph(nrofInputs);
        final OdeHelperBackward.InputArrays inputArrays = getTestInputArrays(nrofInputs, graph);

        final INDArray time = Nd4j.arange(2);
        final OdeHelperBackward helper = new SingleStepAdjoint(
                new DormandPrince54Solver(new SolverConfig(1e-3, 1e-3, 0.1, 10)),
                time, -1);

        Pair<Gradient, INDArray[]> gradients = helper.solve(graph, inputArrays, new OdeHelperBackward.MiscPar(
                false,
                LayerWorkspaceMgr.noWorkspaces(),
                "dummy"));

        final INDArray parGrad = gradients.getFirst().getGradientFor("dummy");
        assertEquals("Incorect gradients view!", graph.getGradientsViewArray(), parGrad);
        assertEquals("Incorrect number of input gradients!", 1, gradients.getSecond().length);

        final INDArray inputGrad = gradients.getSecond()[0];
        assertArrayEquals("Incorrect input gradient shape!", inputArrays.getLastInputs()[0].shape(), inputGrad.shape());

        assertNotEquals("Expected non-zero parameter gradient!", 0.0, parGrad.sumNumber().doubleValue(),1e-10);
        assertNotEquals("Expected non-zero input gradient!", 0.0, inputGrad.sumNumber().doubleValue(), 1e-10);
    }

    /**
     * Smoke test for backwards solve without time gradients
     */
    @Test
    public void solveWithTime() {
        final int nrofInputs = 5;
        final ComputationGraph graph = getTestGraph(nrofInputs);
        final OdeHelperBackward.InputArrays inputArrays = getTestInputArrays(nrofInputs, graph);

        final INDArray time = Nd4j.arange(2);
        final OdeHelperBackward helper = new SingleStepAdjoint(
                new DormandPrince54Solver(new SolverConfig(1e-3, 1e-3, 0.1, 10)),
                time, 1);

        Pair<Gradient, INDArray[]> gradients = helper.solve(graph, inputArrays, new OdeHelperBackward.MiscPar(
                false,
                LayerWorkspaceMgr.noWorkspaces(),
                "dummy"));

        final INDArray parGrad = gradients.getFirst().getGradientFor("dummy");
        assertEquals("Incorect gradients view!", graph.getGradientsViewArray(), parGrad);
        assertEquals("Incorrect number of input gradients!", 2, gradients.getSecond().length);

        final INDArray inputGrad = gradients.getSecond()[0];
        final INDArray timeGrad = gradients.getSecond()[1];
        assertArrayEquals("Incorrect input gradient shape!", inputArrays.getLastInputs()[0].shape(), inputGrad.shape());
        assertArrayEquals("Incorrect time gradient shape!", time.shape(), timeGrad.shape());

        assertNotEquals("Expected non-zero parameter gradient!", 0.0, parGrad.sumNumber().doubleValue(),1e-10);
        assertNotEquals("Expected non-zero input gradient!", 0.0, inputGrad.sumNumber().doubleValue(), 1e-10);
        assertNotEquals("Expected non-zero time gradient!", 0.0, timeGrad.sumNumber().doubleValue(), 1e-10);
    }

    @NotNull
    private static OdeHelperBackward.InputArrays getTestInputArrays(int nrofInputs, ComputationGraph graph) {
        final INDArray input = Nd4j.arange(nrofInputs);
        final INDArray output = input.add(1);
        final INDArray epsilon = Nd4j.ones(nrofInputs).assign(0.01);
        final NonContiguous1DView realGrads = new NonContiguous1DView();
        realGrads.addView(graph.getGradientsViewArray());

        return new OdeHelperBackward.InputArrays(
                new INDArray[]{input},
                output,
                epsilon,
                realGrads
        );
    }

    /**
     *
     * Create a simple graph for testing
     * @param nrofInputs Determines the number of inputs (nIn) to the graph
     * @return a {@link ComputationGraph}
     */
    @NotNull
    static ComputationGraph getTestGraph(int nrofInputs) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(0.01))
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofInputs))
                .addInputs("input")
                .addLayer("dense", new DenseLayer.Builder().nOut(nrofInputs).activation(new ActivationIdentity()).build(), "input")
                .allowNoOutput(true)
                .build());
        graph.init();
        graph.initGradientsView();
        return graph;
    }
}
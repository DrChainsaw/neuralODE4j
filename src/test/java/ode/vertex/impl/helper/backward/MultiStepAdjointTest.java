package ode.vertex.impl.helper.backward;

import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import ode.vertex.impl.NonContiguous1DView;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link MultiStepAdjoint}
 *
 * @author Christian Skarby
 */
public class MultiStepAdjointTest {

    /**
     * Smoke test for backwards solve without time gradients
     */
    @Test
    public void solveNoTime() {
        final int nrofInputs = 7;
        final int nrofTimeSteps = 5;
        final ComputationGraph graph = SingleStepAdjointTest.getTestGraph(nrofInputs);
        final OdeHelperBackward.InputArrays inputArrays = getTestInputArrays(nrofInputs, nrofTimeSteps, graph);

        final INDArray time = Nd4j.arange(nrofTimeSteps);
        final OdeHelperBackward helper = new MultiStepAdjoint(
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

        assertNotEquals("Expected non-zero parameter gradient!", 0.0, parGrad.sumNumber().doubleValue(), 1e-10);
        assertNotEquals("Expected non-zero input gradient!", 0.0, inputGrad.sumNumber().doubleValue(), 1e-10);
    }

    /**
     * Smoke test for backwards solve without time gradients
     */
    @Test
    public void solveWithTime() {
        final int nrofInputs = 5;
        final int nrofTimeSteps = 7;
        final ComputationGraph graph = SingleStepAdjointTest.getTestGraph(nrofInputs);
        final OdeHelperBackward.InputArrays inputArrays = getTestInputArrays(nrofInputs, nrofTimeSteps, graph);

        final INDArray time = Nd4j.arange(nrofTimeSteps);
        final OdeHelperBackward helper = new MultiStepAdjoint(
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
    private static OdeHelperBackward.InputArrays getTestInputArrays(int nrofInputs, int nrofTimeSteps, ComputationGraph graph) {
        final int batchSize = 3;
        final INDArray input = Nd4j.arange(batchSize * nrofInputs).reshape(batchSize, nrofInputs);
        final INDArray output = Nd4j.arange(batchSize * nrofInputs * nrofTimeSteps).reshape(batchSize, nrofInputs, nrofTimeSteps);
        final INDArray epsilon = Nd4j.ones(batchSize, nrofInputs, nrofTimeSteps);
        final NonContiguous1DView realGrads = new NonContiguous1DView();
        realGrads.addView(graph.getGradientsViewArray());

        return new OdeHelperBackward.InputArrays(
                new INDArray[]{input},
                output,
                epsilon,
                realGrads
        );
    }
}
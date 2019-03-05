package ode.vertex.impl.helper.backward;

import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.helper.InputStep;
import ode.vertex.impl.NonContiguous1DView;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
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

        assertNotEquals("Expected non-zero parameter gradient!", 0.0, parGrad.sumNumber().doubleValue(), 1e-10);
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

    /**
     * Test the result vs the result from the original repo. Reimplementation of test_adjoint in
     * https://github.com/rtqichen/torchdiffeq/blob/master/tests/gradient_tests.py.
     * <br><br>
     * Numbers below from the following test ODE:
     * <br><br>
     * <pre>
     * class Linear1DODE(torch.nn.Module):
     *
     *     def __init__(self, device):
     *         super(Linear1DODE, self).__init__()
     *         self.a = torch.nn.Parameter(torch.tensor(0.2).to(device))
     *         self.b = torch.nn.Parameter(torch.tensor(3.0).to(device))
     *
     *     def forward(self, t, y):
     *         return self.a * y + self.b
     *
     *     def y_exact(self, t):
     *         return t
     * </pre>
     */
    @Test
    public void testGradVsReferenceLinear1dOde() {

        final long nrofTimeSteps = 10;
        final long nrofDims = 1;

        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofDims), InputType.feedForward(nrofTimeSteps));

        String next = "y0";
        builder.addInputs(next, "time");

        builder.addVertex("ode", new OdeVertex.Builder("0", new DenseLayer.Builder()
                .activation(new ActivationIdentity())
                .weightInit(new ConstantDistribution(0.2))
                .biasInit(3)
                .nOut(nrofDims)
                .build())
                .odeConf(new InputStep(
                        new ode.solve.conf.DormandPrince54Solver(
                                new SolverConfig(1e-12, 1e-6, 1e-20,1e2)),
                        1, true))
                .build(), next, "time");

        builder.allowNoOutput(true);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        graph.initGradientsView();

        final INDArray time = Nd4j.linspace(1, 8, nrofTimeSteps);
        final INDArray y0 = time.getScalar(0).dup().reshape(1,1);

        final GraphVertex odevert = graph.getVertex("ode");

        odevert.setInputs(y0, time);
        final INDArray ys = odevert.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable());

        // From original repo
        final double[] expectedYs = {1.0000,  3.6929,  6.8391, 10.5147, 14.8090, 19.8261, 25.6876, 32.5355,
                40.5361, 49.8832};
        assertArrayEquals("Incorrect ys: ", expectedYs, ys.reshape(nrofTimeSteps).toDoubleVector(), 1e-3);

        final INDArray lossgrad = Nd4j.linspace(0, nrofTimeSteps-1, nrofTimeSteps);
        for(int i = 0; i < lossgrad.length()/2; i++) {
            lossgrad.getScalar(i*2).negi();
        }

        odevert.setEpsilon(lossgrad.reshape(ys.shape()));
        Pair<Gradient, INDArray[]> grads = odevert.doBackward(false, LayerWorkspaceMgr.noWorkspacesImmutable());

        final double expectedYsGrad = 20.9211;
        assertEquals("Incorrect loss gradient: ", expectedYsGrad, grads.getSecond()[0].getDouble(0),1e-3);

        final double[] expectedParsGrad = { -1232.8964,    -79.6053}; // Note: Gradients in DL4J have opposite sign compared to pytorch
        assertArrayEquals("Incorrect parameter gradient: ", expectedParsGrad, grads.getFirst().gradient().toDoubleVector(), 1e-3);

        final double[] expectedTimeGrad = {-66.9474,   3.7386,  -8.7356,  15.3088, -23.8472,  34.8261, -48.8251,
                66.5498, -88.8578, 116.7898};
        assertArrayEquals("Incorrect time grad!", expectedTimeGrad, grads.getSecond()[1].toDoubleVector(),1e-3);
    }


}
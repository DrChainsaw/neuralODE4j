package ode.vertex.impl.helper.backward;

import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import ode.vertex.conf.OdeVertex;
import ode.vertex.conf.helper.InputStep;
import ode.vertex.impl.gradview.NonContiguous1DView;
import ode.vertex.impl.helper.NoTimeInput;
import ode.vertex.impl.helper.backward.timegrad.CalcMultiStepTimeGrad;
import ode.vertex.impl.helper.backward.timegrad.NoMultiStepTimeGrad;
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
import org.nd4j.linalg.indexing.NDArrayIndex;
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
                time, NoMultiStepTimeGrad.factory);

        INDArray[] gradients = helper.solve(graph, inputArrays, new OdeHelperBackward.MiscPar(
                false,
                LayerWorkspaceMgr.noWorkspacesImmutable()));

        assertEquals("Incorrect number of input gradients!", 1, gradients.length);

        final INDArray inputGrad = gradients[0];
        assertArrayEquals("Incorrect input gradient shape!", inputArrays.getGraphInputOutput().y0().shape(), inputGrad.shape());

        final INDArray parGrad = graph.getGradientsViewArray();
        assertNotEquals("Expected non-zero parameter gradient!", 0.0, parGrad.maxNumber().doubleValue(), 1e-10);
        assertNotEquals("Expected non-zero input gradient!", 0.0, inputGrad.maxNumber().doubleValue(), 1e-10);
    }

    /**
     * Smoke test for backwards solve without time gradients
     */
    @Test
    public void solveWithTime() throws InterruptedException {
        final int nrofInputs = 5;
        final int nrofTimeSteps = 7;
        final ComputationGraph graph = SingleStepAdjointTest.getTestGraph(nrofInputs);
        final OdeHelperBackward.InputArrays inputArrays = getTestInputArrays(nrofInputs, nrofTimeSteps, graph);

        final INDArray time = Nd4j.arange(nrofTimeSteps);
        final OdeHelperBackward helper = new MultiStepAdjoint(
                new DormandPrince54Solver(new SolverConfig(1e-3, 1e-3, 0.1, 10)),
                time, new CalcMultiStepTimeGrad.Factory(time, 1));

        INDArray[] gradients = helper.solve(graph, inputArrays, new OdeHelperBackward.MiscPar(
                false,
                LayerWorkspaceMgr.noWorkspacesImmutable()));


        assertEquals("Incorrect number of input gradients!", 2, gradients.length);

        final INDArray inputGrad = gradients[0];
        final INDArray timeGrad = gradients[1];
        assertArrayEquals("Incorrect input gradient shape!", inputArrays.getGraphInputOutput().y0().shape(), inputGrad.shape());
        assertArrayEquals("Incorrect time gradient shape!", time.shape(), timeGrad.shape());

        final INDArray parGrad = graph.getGradientsViewArray();
        assertNotEquals("Expected non-zero parameter gradient!", 0.0, parGrad.maxNumber().doubleValue(), 1e-10);
        assertNotEquals("Expected non-zero input gradient!", 0.0, inputGrad.maxNumber().doubleValue(), 1e-10);
        assertNotEquals("Expected non-zero time gradient!", 0.0, timeGrad.maxNumber().doubleValue(), 1e-10);
    }

    @NotNull
    private static OdeHelperBackward.InputArrays getTestInputArrays(int nrofInputs, int nrofTimeSteps, ComputationGraph graph) {
        final int batchSize = 3;
        final INDArray input = Nd4j.arange(batchSize * nrofInputs).reshape(batchSize, nrofInputs);
        final INDArray output = Nd4j.arange(batchSize * nrofInputs * nrofTimeSteps).reshape(batchSize, nrofInputs, nrofTimeSteps);
        final INDArray epsilon = Nd4j.ones(batchSize, nrofInputs, nrofTimeSteps).assign(0.01);
        final NonContiguous1DView realGrads = new NonContiguous1DView();
        realGrads.addView(graph.getGradientsViewArray());

        return new OdeHelperBackward.InputArrays(
                new NoTimeInput(new INDArray[]{input}),
                output,
                epsilon,
                realGrads
        );
    }

    /**
     * Test that an exception is thrown if time array is not sorted
     */
    @Test(expected = IllegalArgumentException.class)
    public void timeNotSorted() {
        final INDArray time = Nd4j.create(new double[]{0, 1, 2, 1.5, 3, 4});
        new MultiStepAdjoint(new DormandPrince54Solver(
                new SolverConfig(1, 1, 1e-1, 1)),
                time, new CalcMultiStepTimeGrad.Factory(time, 1));
    }

    /**
     * Test the result vs the result from the original repo. Reimplementation of test_adjoint in
     * https://github.com/rtqichen/torchdiffeq/blob/master/tests/gradient_tests.py.
     * <br><br>
     * Numbers below from the following test ODE:
     * <br><br>
     * <pre>
     *     class Linear1DODE(torch.nn.Module):
     *
     *     def __init__(self, device):
     *         super(Linear1DODE, self).__init__()
     *         self.a = torch.nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2]).reshape(2,2).to(device))
     *         self.b = torch.nn.Parameter(torch.tensor([3.0, 3.0]).reshape(1,2).to(device))
     *
     *     def forward(self, t, y):
     *         return y.matmul(self.a)+ self.b
     *
     *     def y_exact(self, t):
     *         return t
     * Test code (to set same inputs as below):
     *         f, y0, t_points, _ = construct_problem(TEST_DEVICE, ode='linear1D')
     *
     *         y0 = torch.linspace(-1.23,2.34,4).reshape(2, 2).type_as(y0)
     *         y0.requires_grad = True
     *
     *         print('t: ', t_points)
     *         print('y0: ', y0)
     *
     *         params = (list(f.parameters()))
     *         optimizer = torch.optim.SGD(params, lr=0.0001)
     *
     *
     *         optimizer.zero_grad()
     *         func = lambda y0, t_points: torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
     *         ys = func(y0, t_points).permute(1, 2, 0)
     *
     *         gradys = torch.linspace(3,  13, ys.numel()).type_as(ys).reshape(ys.size())
     *         for i in range (0, int(gradys.size()[0]/2)):
     *             gradys[i*2, :, i*2+1] = -gradys[i*2,:,i*2+1]
     *
     *         ys.backward(gradys)
     * </pre>
     */
    @Test
    public void testGradVsReferenceLinear1dOdeForward() {

        final long nrofTimeSteps = 10;
        final long nrofDims = 2;

        final GraphVertex odevert = createOdeVertex(nrofTimeSteps, nrofDims);

        final INDArray time = Nd4j.linspace(1, 8, nrofTimeSteps);
        final INDArray y0 = Nd4j.linspace(-1.23, 2.34, 2 * nrofDims).reshape(2, nrofDims);

        odevert.setInputs(y0, time);
        final INDArray ys = odevert.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable());

        // From original repo
        final double[][][] expectedYs = {{{-1.23, 1.2753194351313588, 4.694932254437282, 9.36250433337755, 15.73345986285669, 24.429438907671752, 36.29894018833409, 52.50011823313958, 74.61376567400269, 104.79757535834692},
                {-0.040000000000000036, 2.4653194351313603, 5.884932254437281, 10.552504333377543, 16.92345986285671, 25.61943890767168, 37.488940188334105, 53.69011823313958, 75.8037656740027, 105.98757535834685}},
                {{1.15, 4.523878831433274, 9.129023844467975, 15.414778231911933, 23.994455416185012, 35.70520942482531, 51.689701681157864, 73.50760277718506, 103.28774415967294, 143.93586077027197},
                        {2.34, 5.7138788314332745, 10.319023844467976, 16.604778231911936, 25.184455416185003, 36.89520942482521, 52.879701681157854, 74.69760277718505, 104.47774415967294, 145.12586077027197}}};

        for (int i = 0; i < expectedYs.length; i++) {
            for (int j = 0; j < expectedYs.length; j++) {
                assertArrayEquals("Incorrect ys: ",
                        expectedYs[i][j],
                        ys.reshape(ys.size(0), nrofDims, nrofTimeSteps).getRow(i).getRow(j).toDoubleVector(), 1e-3);
            }
        }

        final INDArray lossgrad = Nd4j.linspace(3, 13, ys.length()).reshape(ys.shape());
        for (int i = 0; i < lossgrad.size(0) / 2; i++) {
            lossgrad.get(NDArrayIndex.point(i * 2), NDArrayIndex.all(), NDArrayIndex.point(i * 2 + 1)).negi();
        }

        odevert.setEpsilon(lossgrad.reshape(ys.shape()));
        Pair<Gradient, INDArray[]> grads = odevert.doBackward(false, LayerWorkspaceMgr.noWorkspacesImmutable());

        final double[][] expectedYsGrad = {{330.3413671858052, 350.8541876986256}, {641.5288548171546, 667.1698804581804}};
        for (int i = 0; i < expectedYsGrad.length; i++) {
            assertArrayEquals("Incorrect loss gradient: ",
                    expectedYsGrad[i],
                    grads.getSecond()[0].getRow(i).toDoubleVector(), 1e-3);
        }

        // Note: Order of element 1 and 2 are swapped
        final double[] expectedParsGrad = {26399.28182908193, 28805.83177942673, 29982.48959443847, 32597.88284962647, 2022.3108828170273, 2197.8094583156044};
        // Compare one by one due to large dynamic range
        for (int i = 0; i < expectedParsGrad.length; i++) {
            assertEquals("Incorrect parameter gradient for param " + i + "!",
                    1,
                    grads.getFirst().gradient().getDouble(i) / expectedParsGrad[i], 1e-4);
        }

        // First element from reference implementation sure looks weird...
        final double[] expectedTimeGrad = {-6617.017543489908, 63.564528808863464, 185.79311916695133, 262.00021152296233, 369.0851188922797, 519.4357040639363,
                730.3690674996153, 1026.0795986642863, 1440.3518309918456, 2020.3383638791681};
        // Compare one by one due to large dynamic range
        for (int i = 0; i < expectedTimeGrad.length; i++) {
            assertEquals("Incorrect time gradient for param " + i + "!", 1, grads.getSecond()[1].getDouble(i) / expectedTimeGrad[i], 1e-4);
        }
    }

    /**
     * Same test as above but backwards
     */
    @Test
    public void testGradVsReferenceLinear1dOdeBackward() {

        final long nrofTimeSteps = 10;
        final long nrofDims = 2;

        final GraphVertex odevert = createOdeVertex(nrofTimeSteps, nrofDims);

        final INDArray time = Nd4j.linspace(-1, -8, nrofTimeSteps);
        final INDArray y0 = Nd4j.linspace(-1.23, 2.34, 2 * nrofDims).reshape(2, nrofDims);

        odevert.setInputs(y0, time);
        final INDArray ys = odevert.doForward(true, LayerWorkspaceMgr.noWorkspacesImmutable());

        // From original repo
        final double[][][] expectedYs = {{{-1.23, -3.0654783048539187, -4.410209436983333, -5.395402672421891, -6.117186729574334, -6.645989569834943, -7.033407758920375, -7.317242594011386, -7.525189149078903, -7.677538128370835},
                {-0.040000000000000036, -1.875478304853917, -3.2202094369833336, -4.205402672421891, -4.927186729574334, -5.4559895698349425, -5.843407758920375, -6.127242594011386, -6.335189149078852, -6.487538128370829}},
                {{1.15, -1.321813099544714, -3.132743808435675, -4.459489833436336, -5.4315063823619125, -6.143637811088716, -6.665368496900053, -7.047604920850001, -7.327643653785053, -7.532809904849017},
                        {2.34, -0.13181309954471376, -1.9427438084356758, -3.2694898334363294, -4.241506382361924, -4.953637811088717, -5.475368496900053, -5.857604920850002, -6.137643653785053, -6.342809904849003}}};

        for (int i = 0; i < expectedYs.length; i++) {
            for (int j = 0; j < expectedYs.length; j++) {
                assertArrayEquals("Incorrect ys: ",
                        expectedYs[i][j],
                        ys.reshape(ys.size(0), nrofDims, nrofTimeSteps).getRow(i).getRow(j).toDoubleVector(), 1e-3);
            }
        }

        final INDArray lossgrad = Nd4j.linspace(3, 13, ys.length()).reshape(ys.shape());
        for (int i = 0; i < lossgrad.size(0) / 2; i++) {
            lossgrad.get(NDArrayIndex.point(i * 2), NDArrayIndex.all(), NDArrayIndex.point(i * 2 + 1)).negi();
        }

        odevert.setEpsilon(lossgrad.reshape(ys.shape()));
        Pair<Gradient, INDArray[]> grads = odevert.doBackward(false, LayerWorkspaceMgr.noWorkspacesImmutable());

        final double[][] expectedYsGrad = {{0.4791608727137171, 20.99198138553419}, {22.890949753489018, 48.53197539451459}};
        for (int i = 0; i < expectedYsGrad.length; i++) {
            assertArrayEquals("Incorrect loss gradient: ",
                    expectedYsGrad[i],
                    grads.getSecond()[0].getRow(i).toDoubleVector(), 1e-3);
        }

        // Note: Order of element 1 and 2 are swapped
        final double[] expectedParsGrad = {902.9542328576254, 696.559656699922, 1683.4912664748028, 1268.25338547379, -173.44082030059053, -348.93939579916616};
        // Compare one by one due to large dynamic range
        for (int i = 0; i < expectedParsGrad.length; i++) {
            assertEquals("Incorrect parameter gradient for param " + i + "!",
                    1,
                    grads.getFirst().gradient().getDouble(i) / expectedParsGrad[i], 1e-4);
        }

        // First element from reference implementation sure looks weird...
        final double[] expectedTimeGrad = {-229.96656912353825, 34.11827941396485, 53.52715859322072, 40.515245326331545, 30.634856397062812, 23.14160607429379,
                17.465314677923114, 13.17005445759693, 9.923109065345606, 7.470945117798939};
        // Compare one by one due to large dynamic range
        for (int i = 0; i < expectedTimeGrad.length; i++) {
            assertEquals("Incorrect time gradient for param " + i + "!", 1, grads.getSecond()[1].getDouble(i) / expectedTimeGrad[i], 1e-4);
        }
    }

    GraphVertex createOdeVertex(long nrofTimeSteps, long nrofDims) {
        final ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(666)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofDims), InputType.feedForward(nrofTimeSteps));

        String next = "y0";
        builder.addInputs(next, "time");

        builder.addVertex("ode", new OdeVertex.Builder(
                builder.getGlobalConfiguration(), "0", new DenseLayer.Builder()
                .activation(new ActivationIdentity())
                .weightInit(new ConstantDistribution(0.2))
                .biasInit(3)
                .nOut(nrofDims)
                .build())
                .odeConf(new InputStep(
                        new ode.solve.conf.DormandPrince54Solver(
                                new SolverConfig(1e-12, 1e-6, 1e-20, 1e2)),
                        1, true, true))
                .build(), next, "time");

        builder.allowNoOutput(true);

        final ComputationGraph graph = new ComputationGraph(builder.build());
        graph.init();
        graph.initGradientsView();

        return graph.getVertex("ode");
    }
}
package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderSolver;
import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link SingleStepOdeHelperForward}
 *
 * @author Christian Skarby
 */
public class SingleStepOdeHelperForwardTest {

    /**
     * Test if a simple ODE can be solved
     */
    @Test
    public void solve() {
        final double exponent = 2;
        final int nrofInputs = 7;
        final ComputationGraph graph = getSimpleExpGraph(exponent, nrofInputs);

        final INDArray input = Nd4j.arange(nrofInputs);
        final INDArray expected = input.mul(Math.exp(2));

        final FirstOrderSolver solver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 100));
        final OdeHelperForward helper = new SingleStepOdeHelperForward(
                solver,
                Nd4j.linspace(0, 1, 2));

        final INDArray actual = helper.solve(graph, LayerWorkspaceMgr.noWorkspaces(), new INDArray[]{input});

        assertArrayEquals("Incorrect answer!", expected.toDoubleVector(),actual.toDoubleVector(), 1e-3);
    }

    /**
     * dy/dt = exponent*y => y = y0*e^(exponent*t)
     * @param exponent Exponent of e
     * @param nrofInputs Number of elements in y
     * @return a {@link ComputationGraph}
     */
    @NotNull
    static ComputationGraph getSimpleExpGraph(double exponent, int nrofInputs) {
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofInputs))
                .addInputs("input")

                .addVertex("scale", new ScaleVertex(exponent), "input")
                .allowNoOutput(true)
                .build());
        graph.init();
        return graph;
    }
}
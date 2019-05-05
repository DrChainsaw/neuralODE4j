package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderSolver;
import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import ode.vertex.impl.helper.NoTimeInput;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link InputStep}. Also tests {@link FixedStep} as {@link InputStep} depends on it
 *
 * @author Christian Skarby
 */
public class InputStepTest {

    /**
     * Test if a simple ODE can be solved
     */
    @Test
    public void solveSingleStep() {
        final double exponent = 2;
        final int nrofInputs = 7;
        final ComputationGraph graph = SingleStepTest.getSimpleExpGraph(exponent, nrofInputs);

        final INDArray input = Nd4j.arange(nrofInputs);
        final INDArray expected = input.mul(Math.exp(2));

        final FirstOrderSolver solver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 100));
        final OdeHelperForward helper = new InputStep(
                solver,
                1, false);

        final INDArray actual = helper.solve(graph, LayerWorkspaceMgr.noWorkspaces(), new NoTimeInput(new INDArray[]{input, Nd4j.linspace(0, 1, 2)}));

        assertArrayEquals("Incorrect answer!", expected.toDoubleVector(),actual.toDoubleVector(), 1e-3);
    }


    /**
     * Test if a simple ODE can be solved
     */
    @Test
    public void solveMultiStep() {
        final INDArray t = Nd4j.arange(5).castTo(Nd4j.defaultFloatingPointType());
        final double exponent = 0.12;
        final int nrofInputs = 4;
        final ComputationGraph graph = SingleStepTest.getSimpleExpGraph(exponent, nrofInputs);

        final INDArray input = Nd4j.arange(nrofInputs).reshape(1, nrofInputs).castTo(Nd4j.defaultFloatingPointType());
        final INDArray expected = input.transpose().mmul(Transforms.exp(t.mul(exponent)).reshape(1, 5));

        final FirstOrderSolver solver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 100));
        final OdeHelperForward helper = new InputStep(
                solver,
                1, false);

        final INDArray actual = helper.solve(graph, LayerWorkspaceMgr.noWorkspaces(), new NoTimeInput(new INDArray[]{input, t})).reshape(expected.shape());

        for(int row = 0; row < actual.rows(); row++) {
            assertArrayEquals("Incorrect answer!", expected.getRow(row).toDoubleVector(), actual.getRow(row).toDoubleVector(), 1e-3);
        }
    }
}
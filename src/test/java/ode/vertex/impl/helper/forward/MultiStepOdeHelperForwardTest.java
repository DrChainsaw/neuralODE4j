package ode.vertex.impl.helper.forward;

import ode.solve.api.FirstOrderSolver;
import ode.solve.conf.SolverConfig;
import ode.solve.impl.DormandPrince54Solver;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link MultiStepOdeHelperForward}
 *
 * @author Christian Skarby
 */
public class MultiStepOdeHelperForwardTest {

    /**
     * Test if a simple ODE can be solved
     */
    @Test
    public void solve() {
        final INDArray t = Nd4j.arange(4);
        final double exponent = 1.23;
        final int nrofInputs = 3;
        final ComputationGraph graph = SingleStepOdeHelperForwardTest.getSimpleExpGraph(exponent, nrofInputs);

        final INDArray input = Nd4j.arange(nrofInputs);
        final INDArray expected = input.transpose().mmul(Transforms.exp(t.mul(exponent)));

        final FirstOrderSolver solver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 100));
        final OdeHelperForward helper = new MultiStepOdeHelperForward(
                solver,
                t);

        final INDArray actual = helper.solve(graph, LayerWorkspaceMgr.noWorkspaces(), new INDArray[]{input}).reshape(expected.shape());

        for(int row = 0; row < actual.rows(); row++) {
            assertArrayEquals("Incorrect answer!", expected.getRow(row).toDoubleVector(), actual.getRow(row).toDoubleVector(), 1e-3);
        }
    }
}
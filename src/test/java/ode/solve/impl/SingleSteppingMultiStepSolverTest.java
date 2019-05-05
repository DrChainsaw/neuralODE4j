package ode.solve.impl;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.conf.SolverConfig;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link SingleSteppingMultiStepSolver}
 *
 * @author Christian Skarby
 */
public class SingleSteppingMultiStepSolverTest {

    /**
     * Test that solving {@link ode.solve.CircleODE} in multiple steps gives the same result as if all steps are done
     * in one solve.
     */
    @Test
    public void integrate() {
        final int nrofSteps = 20;
        final FirstOrderEquation circle = new CircleODE(new double[] {1.23, 4.56}, 1);
        final FirstOrderSolver actualSolver = new DormandPrince54Solver(new SolverConfig(1e-10, 1e-10, 1e-10, 10));

        final INDArray t = Nd4j.linspace(0, Math.PI, nrofSteps, Nd4j.defaultFloatingPointType()).reshape(1, nrofSteps);
        final INDArray y0 = Nd4j.create(new float[] {0, 0});
        final INDArray ySingle = y0.dup();
        final INDArray yMulti = Nd4j.repeat(ySingle,nrofSteps-1).reshape(nrofSteps-1, y0.length());

        actualSolver.integrate(circle, t.getColumns(0, nrofSteps-1), y0, ySingle);
        new SingleSteppingMultiStepSolver(actualSolver).integrate(circle, t, y0, yMulti);

        assertEquals("Incorrect solution!", ySingle, yMulti.getRow(nrofSteps-2));
    }
}
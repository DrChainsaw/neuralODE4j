package ode.solve.impl;

import ode.solve.CircleODE;
import ode.solve.api.FirstOrderSolver;
import ode.solve.commons.FirstOrderSolverAdapter;
import ode.solve.impl.util.SolverConfig;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link DormandPrince54Solver}
 *
 * @author Christian Skarby
 */
public class DormandPrince54SolverTest {

    /**
     * Test that result is the same as the reference implementation for the {@link CircleODE} problem
     */
    @Test
    public void solveCircle() {
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 0.666);
        final FirstOrderSolver reference = new FirstOrderSolverAdapter(
                new DormandPrince54Integrator(1e-10, 100, 1e-10, 1e-10));
        final FirstOrderSolver test =
                new DormandPrince54Solver(new SolverConfig(
                        1e-10, 1e-10, 1e-10, 100));

        final INDArray y0 = Nd4j.create(new double[]{3, 5});
        final INDArray ts = Nd4j.create(new double[]{7, 11});
        final INDArray y = Nd4j.create(1, 2);

        assertEquals("Incorrect solution!", reference.integrate(equation, ts, y0, y), test.integrate(equation, ts, y0, y));
    }
}
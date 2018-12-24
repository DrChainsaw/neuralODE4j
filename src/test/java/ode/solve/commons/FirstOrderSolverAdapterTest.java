package ode.solve.commons;

import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link FirstOrderSolverAdapter}
 *
 * @author Christian Skarby
 */
public class FirstOrderSolverAdapterTest {

    /**
     * Test that solution for {@link CircleODE} is the same using {@link DormandPrince54Integrator} and the wrapped version of the same instance.
     */
    @Test
    public void solveCircle() {
        final FirstOrderIntegrator solver = new DormandPrince54Integrator(1e-10, 100, 1e-10, 1e-10);
        final CircleODE equation = new CircleODE(new double[] {1.23, 4.56}, 0.666);

        final double[] y0 = {3, 5};
        final double[] ts = {7, 11};
        final double[] y = new double[2];
        solver.integrate(equation, ts[0], y0, ts[1], y);

        final INDArray actual = new FirstOrderSolverAdapter(solver).integrate(equation, Nd4j.create(ts), Nd4j.create(y0), Nd4j.create(1, 2));

        assertEquals("Incorrect answer!", Nd4j.create(y), actual);
    }

}
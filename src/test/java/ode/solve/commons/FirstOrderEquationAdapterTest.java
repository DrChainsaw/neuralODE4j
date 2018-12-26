package ode.solve.commons;

import ode.solve.CircleODE;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link FirstOrderEquationAdapter}
 *
 * @author Christian Skarby
 */
public class FirstOrderEquationAdapterTest {

    /**
     * Test that the same results are produced by {@link CircleODE} and the wrapped version of the same instance
     */
    @Test
    public void computeDerivatives() {
        final CircleODE equation = new CircleODE(new double[]{12.34, 45.67}, 0.666);

        double t =1.23;
        double[] y = {12, 13};
        double[] yExpected = new double[2];
        double[] yActual = new double[2];

        equation.computeDerivatives(t, y, yExpected);
        new FirstOrderEquationAdapter(Nd4j.create(y), equation).computeDerivatives(t, y, yActual);

        assertArrayEquals("Incorrect value!", yExpected,
               yActual, 1e-5);
    }
}
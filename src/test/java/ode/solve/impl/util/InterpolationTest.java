package ode.solve.impl.util;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Interpolation}
 *
 * @author Christian Skarby
 */
public class InterpolationTest {

    /**
     * Test interpolation by comparing to the result for the same input from original repo
     */
    @Test
    public void interpolate() {
        final long[] shape = {2, 3, 5};
        final long nrofElems = 2 * 3 * 5;
        final INDArray y0 = Nd4j.linspace(-10, 10, nrofElems).reshape(shape);
        final INDArray yMid = Nd4j.linspace(-7, 13, nrofElems).reshape(shape);
        final INDArray y1 = Nd4j.linspace(-12, 7, nrofElems).reshape(shape);
        final INDArray f0 = Nd4j.linspace(2, 5, nrofElems).reshape(shape);
        final INDArray f1 = Nd4j.linspace(-3, -1, nrofElems).reshape(shape);
        final INDArray dt = Nd4j.scalar(1.23);

        final Interpolation interpolation = new Interpolation();

        interpolation.fitCoeffs(y0, y1, yMid, f0, f1, dt);

        final INDArray output = interpolation.interpolate(-2, 3, 2.34);

        // Output from pytorch repo
        final INDArray expected = Nd4j.create(new double[][][]{
                {{-10.8218, -10.1690, -9.5162, -8.8633, -8.2105},
                        {-7.5577, -6.9049, -6.2521, -5.5993, -4.9465},
                        {-4.2936, -3.6408, -2.9880, -2.3352, -1.6824}},
                {{-1.0296, -0.3768, 0.2760, 0.9288, 1.5817},
                        {2.2345, 2.8873, 3.5401, 4.1929, 4.8457},
                        {5.4985, 6.1513, 6.8042, 7.4570, 8.1098}}
        });

        assertEquals("Incorrect output!", output.toString(), expected.toString());
    }
}
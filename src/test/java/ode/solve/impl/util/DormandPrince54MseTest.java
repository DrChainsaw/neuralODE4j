package ode.solve.impl.util;

import ode.solve.impl.DormandPrince54Solver;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link DormandPrince54Solver.DormandPrince54Mse}
 *
 * @author Christian Skarby
 */
public class DormandPrince54MseTest {

    /**
     * Test MSE estimation
     */
    @Test
    public void estimateMseForward() {
        final INDArray h = Nd4j.create(new double[]{1.23});
        testEstimateError(h);
    }

    /**
     * Test MSE estimation
     */
    @Test
    public void estimateMseBackward() {
        final INDArray h = Nd4j.create(new double[]{-1.23});
        testEstimateError(h);
    }

    private void testEstimateError(INDArray h) {
        final long dim = 100;
        final INDArray yDotK = Nd4j.linspace(-10, 10, 7 * dim).reshape(new long[]{7, dim});
        final INDArray y0 = Nd4j.linspace(-2.34, 3.45, dim, DataType.FLOAT);
        final INDArray y1 = Nd4j.reverse(Nd4j.linspace(-3.45, 4.56, dim, DataType.FLOAT));

        final double minStep = 1e-10;
        final double maxStep = 1e20;
        final double absTol = 2.34e-3;
        final double relTol = 3.45e-5;

        final double expected = new DormandPrince54Integrator(minStep, maxStep, absTol, relTol) {

            double estError(final double[][] yDotK,
                            final double[] y0, final double[] y1,
                            final double h) {
                mainSetDimension = (int) dim;
                return estimateError(yDotK, y0, y1, h);
            }
        }.estError(yDotK.toDoubleMatrix(), y0.toDoubleVector(), y1.toDoubleVector(), h.getDouble(0));

        final INDArray actual = new DormandPrince54Solver.DormandPrince54Mse(
                new SolverConfigINDArray(absTol, relTol, minStep, maxStep))
                .estimateMse(yDotK, y0, y1, h);

        assertEquals("Incorrect MSE!", expected, actual.getDouble(0), 1e-4);
    }
}
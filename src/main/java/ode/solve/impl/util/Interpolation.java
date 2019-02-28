package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Performs interpolation using a fourth order polynomial. Useful for reducing the number of function evaluations
 * when multiple (closely spaced) time steps are used.
 * Reimplementation of https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/interp.py
 *
 * @author Christian Skarby
 */
public class Interpolation {

    private final INDArray[] coeffs = new INDArray[5];

    private void initCoeffs(long[] shape) {
        for(int i = 0; i < coeffs.length; i++) {
            coeffs[i] = Nd4j.createUninitialized(shape);
        }
    }

    /**
     * Fit coefficients for a fourth order polynomial: p = coeffs[0] * x^4 + coeffs[1] * x^3 + coeffs[2] * x^2 + coeffs[3] * x + coeffs[4]
     * @param y0 Function value at start of interval
     * @param y1 Function value at end of interval
     * @param yMid Function value at midpoint of interval
     * @param f0 Derivative value at start of interval
     * @param f1 Derivative value at end of interval
     * @param dt Time between start and end of interval
     */
    public void fitCoeffs(INDArray y0, INDArray y1, INDArray yMid, INDArray f0, INDArray f1, INDArray dt) {
        if(coeffs[0] == null) {
            initCoeffs(y0.shape());
        }

        final INDArray[] inputs = {f0, f1, y0, y1, yMid};

        final double dtdub = dt.getDouble(0);
        dotProduct(coeffs[0], new double[] {-2 * dtdub, 2 * dtdub, -8, -8, 16}, inputs);

        dotProduct(coeffs[1], new double[] {5 * dtdub, -3 * dtdub, 18, 14, -32}, inputs);

        dotProduct(coeffs[2], new double[] {-4 * dtdub, dtdub, -11, -5, 16}, inputs);

        coeffs[3] = f0.mul(dt);
        coeffs[4] = y0.dup();
    }

    private INDArray dotProduct(INDArray output, double[] factors, INDArray[] inputs) {
        output.assign((inputs[0].mul(factors[0])));
        for(int i = 1; i < inputs.length; i++) {
            output.addi(inputs[i].mul(factors[i]));
        }
        output.mul(3);
        return output;
    }

    /**
     * Evaluate interpolation of a fourth order polynomial: p = coeffs[0] * x^4 + coeffs[1] * x^3 + coeffs[2] * x^2 + coeffs[3] * x + coeffs[4]
     * @param t0 Start of interval
     * @param t1 End of interval
     * @param t Wanted time
     * @return Result of the interpolation
     */
    public INDArray interpolate(double t0, double t1, double t) {

        if(t0 > t || t > t1) {
            throw new IllegalArgumentException("t0 < t < t1 not satisfied! t0: " + t0 + ", t: " + t + ", t1: " + t1);
        }

        final double x = ((t - t0) / (t1 - t0));

        // Create array {x^4, x^3, x^2, x, 1};
        final double[] xs = new double[coeffs.length];
        xs[coeffs.length-1] = 1;
        for(int i = coeffs.length-2; i > -1; i--) {
            xs[i] = xs[i+1]*x;
        }

        return dotProduct(Nd4j.createUninitialized(coeffs[0].shape()), xs, coeffs);
    }

}

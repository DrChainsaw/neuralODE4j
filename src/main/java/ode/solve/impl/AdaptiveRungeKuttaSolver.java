package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.util.ButcherTableu;
import ode.solve.impl.util.FirstOrderEquationWithState;
import ode.solve.impl.util.SolverConfig;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Generic adaptive step size Runge-Kutta solver.
 *
 * @author Christian Skarby
 */
public class AdaptiveRungeKuttaSolver implements FirstOrderSolver {

    private final static INDArray MIN_H = Nd4j.create(new double[]{1e-6});

    private final SolverConfig config;
    private final ButcherTableu tableu;
    private final int order;
    private final MseComputation mseComputation;

    public AdaptiveRungeKuttaSolver(SolverConfig config, ButcherTableu tableu, int order, MseComputation mseComputation) {
        this.config = config;
        this.tableu = tableu;
        this.order = order;
        this.mseComputation = mseComputation;
    }

    /**
     * Interface for computing errors
     */
    public interface MseComputation {

        /**
         * Estimate mean square error from the given state
         *
         * @param yDotK
         * @param y0
         * @param y1
         * @param h
         * @return the (scalar) mean square error
         */
        INDArray estimateMse(
                final INDArray yDotK,
                final INDArray y0,
                final INDArray y1,
                final INDArray h
        );
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        if (t.length() != 2) {
            throw new IllegalArgumentException("time must be size 2! Was: " + t);
        }

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(DormandPrince54Solver.class.getSimpleName())) {

            final FirstOrderEquationWithState equationState = new FirstOrderEquationWithState(
                    equation,
                    t.getScalar(0).dup(),
                    y0,
                    order);

            solve(equationState, t);

            return yOut;
        }
    }

    private void solve(FirstOrderEquationWithState equation, INDArray t) {

        boolean isLastStep = false;
        equation.calculateDerivative(0);

        final INDArray hNew = initializeStep(equation, t);


//        // main integration loop
//        do {
//
//            // interpolator.shift();
//
//            // iterate over step size, ensuring local normalized error is smaller than 1
//            double error = 10;
//            while (error >= 1.0) {
//
//
//                stepSize = hNew;
//                if (forward) {
//                    if (stepStart + stepSize >= t) {
//                        stepSize = t - stepStart;
//                    }
//                } else {
//                    if (stepStart + stepSize <= t) {
//                        stepSize = t - stepStart;
//                    }
//                }
//
//                // next stages
//                for (int k = 1; k < stages; ++k) {
//
//                    for (int j = 0; j < y0.length; ++j) {
//                        double sum = a[k - 1][0] * yDotK[0][j];
//                        for (int l = 1; l < k; ++l) {
//                            sum += a[k - 1][l] * yDotK[l][j];
//                        }
//                        yTmp[j] = y[j] + stepSize * sum;
//                    }
//
//                    computeDerivatives(stepStart + c[k - 1] * stepSize, yTmp, yDotK[k]);
//
//                }
//
//                // estimate the state at the end of the step
//                for (int j = 0; j < y0.length; ++j) {
//                    double sum = b[0] * yDotK[0][j];
//                    for (int l = 1; l < stages; ++l) {
//                        sum += b[l] * yDotK[l][j];
//                    }
//                    yTmp[j] = y[j] + stepSize * sum;
//                }
//
//                // estimate the error at the end of the step
//                error = estimateError(yDotK, y, yTmp, stepSize);
//                if (error >= 1.0) {
//                    // reject the step and attempt to reduce error by stepsize control
//                    final double factor =
//                            FastMath.min(maxGrowth,
//                                    FastMath.max(minReduction, safety * FastMath.pow(error, exp)));
//                    hNew = filterStep(stepSize * factor, forward, false);
//                }
//
//            }
//
//            // local error is small enough: accept the step, trigger events and step handlers
//            interpolator.storeTime(stepStart + stepSize);
//            System.arraycopy(yTmp, 0, y, 0, y0.length);
//            System.arraycopy(yDotK[stages - 1], 0, yDotTmp, 0, y0.length);
//            stepStart = acceptStep(interpolator, y, yDotTmp, t);
//            System.arraycopy(y, 0, yTmp, 0, y.length);
//
//            if (!isLastStep) {
//
//                // prepare next step
//                interpolator.storeTime(stepStart);
//
//                if (fsal) {
//                    // save the last evaluation for the next step
//                    System.arraycopy(yDotTmp, 0, yDotK[0], 0, y0.length);
//                }
//
//                // stepsize control for next step
//                final double factor =
//                        FastMath.min(maxGrowth, FastMath.max(minReduction, safety * FastMath.pow(error, exp)));
//                final double scaledH = stepSize * factor;
//                final double nextT = stepStart + scaledH;
//                final boolean nextIsLast = forward ? (nextT >= t) : (nextT <= t);
//                hNew = filterStep(scaledH, forward, nextIsLast);
//
//                final double filteredNextT = stepStart + hNew;
//                final boolean filteredNextIsLast = forward ? (filteredNextT >= t) : (filteredNextT <= t);
//                if (filteredNextIsLast) {
//                    hNew = t - stepStart;
//                }
//
//            }
//
//        } while (!isLastStep);
    }

    /**
     * Initialize the integration step. Computes the first two stages (0 and 1) of the provided equation.
     *
     * @param equation {@link FirstOrderEquationWithState} to initialize the step for
     * @param t        start and stop time
     * @return first integration step
     */
    private INDArray initializeStep(FirstOrderEquationWithState equation, INDArray t) {

        equation.calculateDerivative(0);

        final INDArray scal = abs(equation.getCurrentState()).mul(config.getRelTol()).add(config.getAbsTol());
        final INDArray ratio = equation.getCurrentState().div(scal);
        final INDArray yOnScale2 = ratio.muli(ratio).sum();

        // Reuse ratio variable
        ratio.assign(equation.getStateDot(0).div(scal));
        final INDArray yDotOnScale2 = ratio.muli(ratio).sum();

        final INDArray h = ((yOnScale2.getDouble(0) < 1.0e-10) || (yDotOnScale2.getDouble(0) < 1.0e-10)) ?
                MIN_H : sqrt(yOnScale2.divi(yDotOnScale2));

        final boolean backward = t.argMax().getInt(0) == 0;
        if (backward) {
            h.negi();
        }

        equation.step(h, 0);
        equation.calculateDerivative(1);

        // Reuse ratio variable
        ratio.assign(equation.getStateDot(1).sub(equation.getStateDot(0)).divi(scal));
        ratio.muli(ratio);
        final INDArray yDDotOnScale = sqrt(ratio.sum()).divi(h);

        // step size is computed such that
        // h^order * max (||y'/tol||, ||y''/tol||) = 0.01
        final INDArray maxInv2 = max(sqrt(yDotOnScale2), yDDotOnScale);
        final INDArray h1 = maxInv2.getDouble(0) < 1e-15 ?
                max(MIN_H, abs(h).muli(0.001)) :
                pow(maxInv2.rdivi(0.01), 1d / order);

        h.assign(min(abs(h).muli(100), h1));
        h.assign(max(h, abs(t.getColumn(0)).muli(1e-12)));

        h.assign(max(config.getMinStep(), h));
        h.assign(min(config.getMaxStep(), h));

        if (backward) {
            h.negi();
        }

        return h;
    }


    public static void main(String[] args) {
        final INDArray t = Nd4j.create(new double[]{-1.2, 7.4});
        final INDArray y0 = Nd4j.create(new double[]{3, 5});
        final CircleODE equation = new CircleODE(new double[]{1.23, 4.56}, 0.666);

        final double absTol = 1.23e-3;
        final double relTol = 2.34e-5;
        final double[] scale = new double[equation.getDimension()];
        for (int i = 0; i < scale.length; ++i) {
            scale[i] = absTol + relTol * FastMath.abs(y0.getDouble(i));
        }
        System.out.println("scal ref: " + Arrays.toString(scale));
        final double[] y1 = y0.toDoubleVector();
        final double[] yDot = y0.toDoubleVector();
        equation.computeDerivatives(t.getDouble(0), y1, yDot);
        final double[] yDot1 = y0.toDoubleVector();
        final double stepRef = initializeStepRef(equation,
        true, 5, scale, t.getDouble(0), y0.toDoubleVector(), yDot,
        y1, yDot1);

        final INDArray stepAct = new AdaptiveRungeKuttaSolver(
                new SolverConfig(absTol, relTol, 1e-20, 1e20),
                null, 5, null)
        .initializeStep(new FirstOrderEquationWithState(equation, t.getColumn(0), y0, 5), t);

        System.out.println("ref: " + stepRef);
        System.out.println("act: " + stepAct.getDouble(0));
    }


    /**
     * Initialize the integration step.
     *
     * @param forward forward integration indicator
     * @param order   order of the method
     * @param scale   scaling vector for the state vector (can be shorter than state vector)
     * @param t0      start time
     * @param y0      state vector at t0
     * @param yDot0   first time derivative of y0
     * @param y1      work array for a state vector
     * @param yDot1   work array for the first time derivative of y1
     * @return first integration step
     * @throws MaxCountExceededException  if the number of functions evaluations is exceeded
     * @throws DimensionMismatchException if arrays dimensions do not match equations settings
     */
    private static double initializeStepRef(
            final FirstOrderDifferentialEquations equations,
            final boolean forward, final int order, final double[] scale,
            final double t0, final double[] y0, final double[] yDot0,
            final double[] y1, final double[] yDot1)
            throws MaxCountExceededException, DimensionMismatchException {

        // very rough first guess : h = 0.01 * ||y/scale|| / ||y'/scale||
        // this guess will be used to perform an Euler step
        double ratio;
        double yOnScale2 = 0;
        double yDotOnScale2 = 0;
        for (int j = 0; j < scale.length; ++j) {
            ratio = y0[j] / scale[j];
            yOnScale2 += ratio * ratio;
            ratio = yDot0[j] / scale[j];
            yDotOnScale2 += ratio * ratio;
        }
        System.out.println("yd: " + yDotOnScale2 + " y: " + yOnScale2);
        double h = ((yOnScale2 < 1.0e-10) || (yDotOnScale2 < 1.0e-10)) ?
                1.0e-6 : (0.01 * FastMath.sqrt(yOnScale2 / yDotOnScale2));
        if (!forward) {
            h = -h;
        }

        // perform an Euler step using the preceding rough guess
        for (int j = 0; j < y0.length; ++j) {
            y1[j] = y0[j] + h * yDot0[j];
        }
        equations.computeDerivatives(t0 + h, y1, yDot1);

        // estimate the second derivative of the solution
        double yDDotOnScale = 0;
        for (int j = 0; j < scale.length; ++j) {
            ratio = (yDot1[j] - yDot0[j]) / scale[j];
            yDDotOnScale += ratio * ratio;
        }
        yDDotOnScale = FastMath.sqrt(yDDotOnScale) / h;

        // step size is computed such that
        // h^order * max (||y'/tol||, ||y''/tol||) = 0.01
        final double maxInv2 = FastMath.max(FastMath.sqrt(yDotOnScale2), yDDotOnScale);
        final double h1 = (maxInv2 < 1.0e-15) ?
                FastMath.max(1.0e-6, 0.001 * FastMath.abs(h)) :
                FastMath.pow(0.01 / maxInv2, 1.0 / order);
        h = FastMath.min(100.0 * FastMath.abs(h), h1);
        h = FastMath.max(h, 1.0e-12 * FastMath.abs(t0));  // avoids cancellation when computing t1 - t0
        if (!forward) {
            h = -h;
        }

        return h;

    }

    private static class CircleODE implements FirstOrderEquation, FirstOrderDifferentialEquations {

        private double[] c;
        private double omega;

        public CircleODE(double[] c, double omega) {
            this.c = c;
            this.omega = omega;
        }


        @Override
        public int getDimension() {
            return 2;
        }

        @Override
        public void computeDerivatives(double t, double[] y, double[] yDot) {
            yDot[0] = omega * (c[1] - y[1]);
            yDot[1] = omega * (y[0] - c[0]);
        }

        @Override
        public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
            fy.putScalar(0, omega * (c[1] - y.getDouble(1)));
            fy.putScalar(1, omega * (y.getDouble(0) - c[0]));
            return fy;
        }
    }

}

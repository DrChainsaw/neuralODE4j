package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.util.*;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
        final StepPolicy stepPolicy = new AdaptiveRungeKuttaStepPolicy(config, order);

        final INDArray hNew = stepPolicy.initializeStep(equation, t);


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

}

package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import ode.solve.impl.util.AggStepListener;
import ode.solve.impl.util.ButcherTableu;
import ode.solve.impl.util.FirstOrderEquationWithState;
import ode.solve.impl.util.StepPolicy;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Generic adaptive step size Runge-Kutta solver. Implementation based on
 * {@link org.apache.commons.math3.ode.nonstiff.EmbeddedRungeKuttaIntegrator}.
 *
 * @author Christian Skarby
 */
public class AdaptiveRungeKuttaSolver implements FirstOrderSolver {

    final static WorkspaceConfiguration wsConf = WorkspaceConfiguration.builder()
            .overallocationLimit(0.0)
            .policySpill(SpillPolicy.REALLOCATE)
            .build();

    private final ButcherTableu tableu;
    private final StepPolicy stepPolicy;
    private final MseComputation mseComputation;
    private final AggStepListener listener = new AggStepListener();

    public AdaptiveRungeKuttaSolver(ButcherTableu tableu, StepPolicy stepPolicy, MseComputation mseComputation) {
        this.tableu = tableu;
        this.stepPolicy = stepPolicy;
        this.mseComputation = mseComputation;
    }

    /**
     * Interface for computing errors
     */
    public interface MseComputation {

        /**
         * Estimate mean square error from the given state
         *
         * @param yDotK derivatives computed during the first stages
         * @param y0    estimate of the step at the start of the step
         * @param y1    estimate of the step at the end of the step
         * @param h     current step
         * @return error ratio, greater than 1 if step should be rejected
         */
        INDArray estimateMse(
                final INDArray yDotK,
                final INDArray y0,
                final INDArray y1,
                final INDArray h
        );
    }

    private interface TimeLimit {
        boolean isLastStep(INDArray step);
    }

    private static class TimeLimitForwards implements TimeLimit {

        private final double tLast;
        private final INDArray t;

        private TimeLimitForwards(double tLast, INDArray t) {
            this.tLast = tLast;
            this.t = t;
        }

        @Override
        public boolean isLastStep(INDArray step) {
            if (t.add(step).getDouble(0) >= tLast) {
                step.assign(tLast - t.getDouble(0));
                return true;
            }
            return false;
        }
    }

    private static class TimeLimitBackwards implements TimeLimit {

        private final double tLast;
        private final INDArray t;

        private TimeLimitBackwards(double tLast, INDArray t) {
            this.tLast = tLast;
            this.t = t;
        }

        @Override
        public boolean isLastStep(INDArray step) {
            if (t.add(step).getDouble(0) <= tLast) {
                step.assign(tLast - t.getDouble(0));
                return true;
            }
            return false;
        }
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        if (t.length() != 2) {
            throw new IllegalArgumentException("time must be size 2! Was: " + t);
        }

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsConf, this.getClass().getSimpleName())) {

            final FirstOrderEquationWithState equationState = new FirstOrderEquationWithState(
                    equation,
                    t.getScalar(0).dup(),
                    yOut.assign(y0),
                    tableu.cMid);

            listener.begin(t, y0);

            solve(equationState, t);

            listener.done();

            return yOut;
        }
    }

    private void solve(FirstOrderEquationWithState equation, INDArray t) {
        equation.calculateDerivative(0);

        // Alg variable used for error
        final INDArray error = Nd4j.create(1);
        // Alg variable used for new steps
        final INDArray step = stepPolicy.initializeStep(equation, t);

        // Alg variable for where next step starts
        final TimeLimit timeLimit = t.argMax().getInt(0) == 1 ?
                new TimeLimitForwards(t.getDouble(1), equation.time()) :
                new TimeLimitBackwards(t.getDouble(1), equation.time());

        final long stages = tableu.c.length() + 1;

        // main integration loop
        boolean isLastStep;
        do {

            isLastStep = timeLimit.isLastStep(step);
            // next stages
            for (long k = 1; k < stages; ++k) {
                equation.step(tableu.a[(int) k - 1], step);
                equation.calculateDerivative(k);
            }
            // estimate the state at the end of the step
            equation.step(tableu.b, step);

            // estimate the error at the end of the step
            error.assign(equation.estimateError(mseComputation));

            isLastStep &= acceptStep(equation, step, error);

            // Take a new step. Note: Redundant operation if isLastStep is true
            step.assign(stepPolicy.step(step, error));

        } while (!isLastStep);
    }

    private boolean acceptStep(FirstOrderEquationWithState equation, INDArray step, INDArray error) {
        if (error.getDouble(0) < 1.0) {
            // local error is small enough: accept the step,
            equation.update();

            listener.step(equation, step, error);

            equation.shiftDerivative();

            return true;
        }
        return false;
    }

    @Override
    public void addListener(StepListener... listeners) {
        this.listener.addListeners(listeners);
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        this.listener.clearListeners(listeners);
    }
}

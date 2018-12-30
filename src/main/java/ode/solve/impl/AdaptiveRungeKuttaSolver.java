package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.util.ButcherTableu;
import ode.solve.impl.util.FirstOrderEquationWithState;
import ode.solve.impl.util.SolverConfig;
import ode.solve.impl.util.StepPolicy;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Generic adaptive step size Runge-Kutta solver.
 *
 * @author Christian Skarby
 */
public class AdaptiveRungeKuttaSolver implements FirstOrderSolver {

    private final SolverConfig config;
    private final ButcherTableu tableu;
    private final StepPolicy stepPolicy;
    private final MseComputation mseComputation;

    public AdaptiveRungeKuttaSolver(SolverConfig config, ButcherTableu tableu, StepPolicy stepPolicy, MseComputation mseComputation) {
        this.config = config;
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
                    tableu.c.length() + 1);

            solve(equationState, t);

            return yOut;
        }
    }

    private void solve(FirstOrderEquationWithState equation, INDArray t) {

        equation.calculateDerivative(0);

        // Alg variable used for new steps
        final INDArray stepNew = stepPolicy.initializeStep(equation, t);
        final INDArray step = stepNew.dup();
        // Alg variable for where next step starts
        final boolean forward = t.argMax().getInt(0) == 1;

        final long stages = tableu.c.length()+1;


        // main integration loop
        boolean isLastStep;
        do {

            // iterate over step size, ensuring local normalized error is smaller than 1
            INDArray error = Nd4j.create(1).putScalar(0, 10);
            while (error.getDouble(0) >= 1.0) {

                step.assign(stepNew);
                if (forward) {
                    if (equation.time().add(step).getDouble(0) >= t.getDouble(1)) {
                        step.assign(t.getScalar(1).sub(equation.time()));
                    }
                } else {
                    if (equation.time().add(step).getDouble(0) <= t.getDouble(1)) {
                        step.assign(t.getScalar(1).sub(equation.time()));
                    }
                }
                // next stages
                for (long k = 1; k < stages; ++k) {
                    equation.step(tableu.a[(int) k - 1], step);
                    equation.calculateDerivative(k);
                }

                // estimate the state at the end of the step
                equation.step(tableu.b, step);

                // estimate the error at the end of the step
                error.assign(equation.estimateError(mseComputation));
                if (error.getDouble(0) >= 1.0) {
                    if(forward) {
                        stepNew.assign(stepPolicy.stepForward(step, error));
                    } else {
                        stepNew.assign(stepPolicy.stepBackward(step, error));
                    }
                }

            }

            // local error is small enough: accept the step,
            equation.update();
            isLastStep = equation.time().equalsWithEps(t.getScalar(1), config.getMinStep().getDouble(0) / 1e2);

            if (!isLastStep) {
                // Take a new step
                if(forward) {
                    stepNew.assign(stepPolicy.stepForward(step, error));
                } else {
                    stepNew.assign(stepPolicy.stepBackward(step, error));
                }
            }
        } while (!isLastStep);
    }

}

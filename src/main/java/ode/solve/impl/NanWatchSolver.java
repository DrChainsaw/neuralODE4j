package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.IsNaN;

import static org.nd4j.linalg.indexing.BooleanIndexing.or;

/**
 * Performs Nan checks of a wrapped {@link FirstOrderSolver}
 */
public class NanWatchSolver implements FirstOrderSolver {

    private final FirstOrderSolver solver;
    private final Runnable nanAction;

    public NanWatchSolver(FirstOrderSolver solver) {
        this(solver, () -> {
            throw new IllegalStateException("Detected NaN!");
        });
    }

    public NanWatchSolver(FirstOrderSolver solver, Runnable nanAction) {
        this.solver = solver;
        this.nanAction = nanAction;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        return solver.integrate(new NanWatchEquation(equation, nanAction), t, y0, yOut);
    }

    /**
     * Performs Nan checks of a wrapped {@link FirstOrderEquation}
     */
    public static class NanWatchEquation implements FirstOrderEquation {

        private final FirstOrderEquation equation;
        private final Runnable nanAction;
        private final static Condition isNan = new IsNaN();

        public NanWatchEquation(FirstOrderEquation equation, Runnable nanAction) {
            this.equation = equation;
            this.nanAction = nanAction;
        }

        @Override
        public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
            if (or(y, isNan) || or(t, isNan) || or(fy, isNan)) {
                nanAction.run();
            }
            return equation.calculateDerivative(y, t, fy);
        }
    }
}


package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderMultiStepSolver;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * {@link FirstOrderSolver} for multiple time steps. Will invoke the wrapped {@link FirstOrderSolver} multiple times for
 * every pair {@code t[n], t[n+1], 0 < n < t.length()-1} and concatenate the solution.
 *
 * @author Christian Skarby
 */
public class SingleSteppingMultiStepSolver implements FirstOrderMultiStepSolver {

    private final FirstOrderSolver solver;

    public SingleSteppingMultiStepSolver(FirstOrderSolver solver) {
        this.solver = solver;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        if(yOut.size(0) + 1 != t.length()) {
            throw new IllegalArgumentException("yOut must have one element less in first dimension compared to t!");
        }
        final INDArrayIndex[] yOutAccess = NDArrayIndex.allFor(yOut);
        final INDArrayIndex[] tAccess = NDArrayIndex.allFor(t);

        for (int step = 1; step < t.length(); step++) {
            yOutAccess[0] = NDArrayIndex.point(step -2);
            final INDArray yPrev = step == 1 ? y0 : yOut.get(yOutAccess);
            yOutAccess[0] = NDArrayIndex.point(step-1);
            final INDArray yCurr = yOut.get(yOutAccess);
            tAccess[tAccess.length-1] = NDArrayIndex.interval(step - 1, step+1);
            solver.integrate(equation, t.get(tAccess), yPrev, yCurr);
        }

        return yOut;
    }

    @Override
    public void addListener(StepListener... listeners) {
        solver.addListener(listeners);
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        solver.clearListeners(listeners);
    }
}

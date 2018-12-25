package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.IntSupplier;

/**
 * Dummy which does not try to solve anything, it just iterates a given number of times.
 *
 * @author Christian Skarby
 */
public class DummyIteration implements FirstOrderSolver {

    private final IntSupplier nrofIterations;

    public DummyIteration(IntSupplier nrofIterations) {
        this.nrofIterations = nrofIterations;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        final int nrofIters = nrofIterations.getAsInt();
        INDArray next = y0;
        for(int i = 0; i < nrofIters; i++) {
            next = equation.calculateDerivative(next, t.getColumn(0), yOut);
        }
        return yOut;
    }
}

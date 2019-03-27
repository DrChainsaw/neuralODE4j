package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import ode.solve.impl.util.AggStepListener;
import ode.solve.impl.util.StateContainer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.IntSupplier;

/**
 * Dummy which does not try to solve anything, it just iterates a given number of times.
 *
 * @author Christian Skarby
 */
public class DummyIteration implements FirstOrderSolver {

    private final IntSupplier nrofIterations;
    private final AggStepListener listener = new AggStepListener();

    public DummyIteration(IntSupplier nrofIterations) {
        this.nrofIterations = nrofIterations;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        final int nrofIters = nrofIterations.getAsInt();
        INDArray next = y0;

        listener.begin(t, y0);

        for (int i = 0; i < nrofIters; i++) {
            next = equation.calculateDerivative(next, t.getColumn(0), yOut);
            listener.step(
                    new StateContainer(t.getColumn(0),
                            next,
                            Nd4j.zeros(next.shape())),
                    Nd4j.zeros(1), Nd4j.zeros(1));
        }

        listener.done();

        return yOut;
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

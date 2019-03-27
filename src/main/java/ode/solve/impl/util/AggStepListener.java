package ode.solve.impl.util;

import ode.solve.api.StepListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

/**
 * Aggregate of {@link StepListener}s. Main purpose is to avoid having listener loops cluttering up the real code.
 *
 * @author Christian Skarby
 */
public class AggStepListener implements StepListener {

    private final Collection<StepListener> listeners = new ArrayList<>();

    @Override
    public void begin(INDArray t, INDArray y0) {
        for (StepListener listener : listeners) {
            listener.begin(t, y0);
        }
    }

    @Override
    public void step(SolverState solverState, INDArray step, INDArray error) {
        for (StepListener listener : listeners) {
            listener.step(solverState, step, error);
        }
    }

    @Override
    public void done() {
        for (StepListener listener : listeners) {
            listener.done();
        }
    }

    public void addListeners(StepListener... listeners) {
        this.listeners.addAll(Arrays.asList(listeners));
    }

    public void clearListeners(StepListener... listeners) {
        if (listeners == null || listeners.length == 0) {
            this.listeners.clear();
        } else {
            this.listeners.removeAll(Arrays.asList(listeners));
        }
    }

    @Override
    public String toString() {
        return this.getClass().getName();
    }
}

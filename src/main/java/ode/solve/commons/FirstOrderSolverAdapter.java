package ode.solve.commons;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Adapter from {@link FirstOrderSolver} to {@link FirstOrderIntegrator}
 *
 * @author Christian Skarby
 */
public class FirstOrderSolverAdapter implements FirstOrderSolver {

    private final FirstOrderIntegrator wrappedSolver;
    private final Map<StepListener, StepListenerAdapter> listeners = new HashMap<>();

    public FirstOrderSolverAdapter(@JsonProperty("wrappedSolver") FirstOrderIntegrator wrappedSolver) {
        this.wrappedSolver = wrappedSolver;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        final FirstOrderDifferentialEquations wrappedEquations = new FirstOrderEquationAdapter(y0.dup(), equation);
        final double[] y = y0.reshape(1, y0.length()).toDoubleVector();
        wrappedSolver.integrate(wrappedEquations, t.getDouble(0), y, t.getDouble(1), y);

        for(StepListener listener: listeners.keySet()) {
            listener.done();
        }

        for(int i = 0; i < y.length; i++) {
            yOut.putScalar(i, y[i]);
        }
        return yOut;
    }

    @Override
    public void addListener(StepListener... listeners) {
        for(StepListener listener: listeners) {
            final StepListenerAdapter adapter = new StepListenerAdapter(listener);
            wrappedSolver.addStepHandler(adapter);
            this.listeners.put(listener, adapter);
        }
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        if(listeners == null || listeners.length == 0 || this.listeners.values().containsAll(Arrays.asList(listeners))) {
            wrappedSolver.clearStepHandlers();
            this.listeners.keySet().removeAll(Arrays.asList(listeners));
        } else {
            throw new UnsupportedOperationException("Can not remove subset of listeners from " + wrappedSolver.getClass());
        }
    }
}

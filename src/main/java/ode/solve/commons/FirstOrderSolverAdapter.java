package ode.solve.commons;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Adapter from {@link FirstOrderSolver} to {@link FirstOrderIntegrator}
 *
 * @author Christian Skarby
 */
public class FirstOrderSolverAdapter implements FirstOrderSolver {

    private final FirstOrderIntegrator wrappedSolver;

    public FirstOrderSolverAdapter(FirstOrderIntegrator wrappedSolver) {
        this.wrappedSolver = wrappedSolver;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        final FirstOrderDifferentialEquations wrappedEquations = new FirstOrderEquationAdapter(y0.dup(), equation);
        final double[] y = y0.reshape(1, y0.length()).toDoubleVector();
        wrappedSolver.integrate(wrappedEquations, t.getDouble(0), y, t.getDouble(1), y);
        for(int i = 0; i < y.length; i++) {
            yOut.putScalar(i, y[i]);
        }
        return yOut;
    }
}

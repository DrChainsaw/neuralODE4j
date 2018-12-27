package ode.solve.commons;

import ode.solve.api.FirstOrderEquation;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Adapter from {@link FirstOrderEquation} to {@link FirstOrderDifferentialEquations}.
 *
 * @author Christian Skarby
 */
public class FirstOrderEquationAdapter implements FirstOrderDifferentialEquations {

    private final INDArray lastResult;
    private final INDArray t;
    private final FirstOrderEquation wrappedEquation;

    public FirstOrderEquationAdapter(INDArray lastResult, FirstOrderEquation wrappedEquation) {
        this.lastResult = lastResult;
        this.t = Nd4j.create(1);
        this.wrappedEquation = wrappedEquation;
    }

    @Override
    public int getDimension() {
        return (int)lastResult.length();
    }

    @Override
    public void computeDerivatives(double t, double[] y, double[] yDot) throws MaxCountExceededException, DimensionMismatchException {
        this.t.putScalar(0, t);
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(this.getClass().getSimpleName())) {
            // Transfer double[] y to an INDArray
            final INDArray yArr = Nd4j.create(y).reshape(lastResult.shape());

            // Perform desired operation
            wrappedEquation.calculateDerivative(yArr, this.t, lastResult);

            // Transfer result in INDArray lastResult to double[] yOut
            System.arraycopy(lastResult.reshape(1 ,lastResult.length()).toDoubleVector(), 0, yDot, 0, yDot.length);
        }
    }
}

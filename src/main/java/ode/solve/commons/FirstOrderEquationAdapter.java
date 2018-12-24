package ode.solve.commons;

import ode.solve.api.FirstOrderEquation;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
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
        wrappedEquation.calculateDerivate(fromDoubleVec(y, lastResult.shape()), this.t, lastResult);
        for(int i = 0; i < yDot.length; i++) {
            yDot[i] = lastResult.getDouble(i);
        }
    }

    private static INDArray fromDoubleVec(double[] vec, long[] shape) {
        final INDArray input = Nd4j.create(1, vec.length);
        return fromDoubleVec(input, vec).reshape(shape);
    }

    private static INDArray fromDoubleVec(INDArray array, double[] vec) {
        for (int i = 0; i < vec.length; i++) {
            array.putScalar(i, vec[i]);
        }
        return array;
    }
}

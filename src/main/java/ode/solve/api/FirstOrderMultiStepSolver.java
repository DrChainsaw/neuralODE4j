package ode.solve.api;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Marker interface for solvers which can handle multiple time steps (i.e t.length() > 2). Does not add any new
 * functionality, just adds API clarity.
 *
 * @author Christian Skarby
 */
public interface FirstOrderMultiStepSolver extends FirstOrderSolver {

    /**
     * Compute an estimated values of <code>Y(t1), Y(t2), ..., Y(tN)</code> for which
     *
     * @param equation is the function <code>F(Y(t))</code> for which <code>dY/dt = F(Y(t))</code>
     * @param t        is a vector of times <code>t0, t1, t2, ..., tN</code> for which <code>Y(t1), Y(t2), ..., Y(tN)</code> is sought
     * @param y0       is the value of </code>Y(t)<code> at <code>t = t0</code>, i.e. <code>Y(t0)</code>
     * @param yOut     will contain the estimated values of <code>Y(t1), Y(t2), ..., Y(tN)</code>.
     * @return yOut, same instance as input param. Note that yOut does not contain <code>Y(t0)</code> in order to comply
     * with {@link FirstOrderSolver} API in the sense that if <code>t</code> has only two values, only <code>Y(t1)</code> is returned
     */
    @Override
    INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut);
}

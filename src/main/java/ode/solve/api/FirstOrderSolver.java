package ode.solve.api;

import ode.solve.impl.listen.StepListener;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for numerical solvers of {@link FirstOrderEquation}s. Does numerical integration from given start values up
 * to the desired value.
 *
 * @author Christian Skarby
 */
public interface FirstOrderSolver {

    /**
     * Compute estimated value of <code>Y(t1)</code> for which
     * @param equation is the function <code>F(Y(t)</code> for which <code>dY/dt = F(Y(t)</code>
     * @param t is a vector with initial value t0 and desired value t1
     * @param y0 is the value of </code>Y(t)<code> at <code>t = t0</code>
     * @param yOut will contain the estimated value of <code>Y(t1)</code>. Implementations shall assume that y0 and yOut
     *            are the same instance.
     * @return yOut, same instance as input param
     */
    INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut);

    /**
     * Add {@link StepListener}s which will be notified of steps taken
     * @param listeners listeners to add
     */
    void addListener(StepListener... listeners);

    /**
     * Clear the given listeners. Clear all listeners if empty
     * @param listeners listeners to remove
     */
    void clearListeners(StepListener ... listeners);
}

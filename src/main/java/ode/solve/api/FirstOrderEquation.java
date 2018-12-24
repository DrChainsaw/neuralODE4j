package ode.solve.api;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for a first order differential equation. Represents the time derivative dY/dT of some (potentially unknown)
 * function Y where dY/dT only depends on Y and T.
 *
 * @author Christian Skarby
 */
public interface FirstOrderEquation {

    /**
     * Calculate <code>dY/dt = F(Y(t))</code>. In other words, this is the definition of <code>F</code>
     * @param y Value of <code>Y(t)</code> above
     * @param t Value of <code>t</code> above
     * @param fy Value of <code>F(Y(t))</code> above given <code>t</code> and <code>Y(t)</code>
     * @return fy, same instance as input param.
     */
    INDArray calculateDerivate(INDArray y, INDArray t, INDArray fy);
}

package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for determining step size
 */
public interface StepPolicy {

    /**
     * Initialize the integration step. Computes the first two stages (0 and 1) of the provided equation.
     *
     * @param equation {@link FirstOrderEquationWithState} to initialize the step for
     * @param t        start and stop time
     * @return first integration step
     */
    INDArray initializeStep(FirstOrderEquationWithState equation, INDArray t);

    /**
     * Filter the integration step in forward direction.
     *
     * @param step           signed step
     * @return a bounded integration step (scaled step if no bound is reach, or a bounded value)
     */
    INDArray filterStepForward(INDArray step, INDArray error);

    /**
     * Filter the integration step in forward direction.
     *
     * @param step           signed step
     * @return a bounded integration step (scaled step if no bound is reach, or a bounded value)
     */
    INDArray filterStepBackward(INDArray step, INDArray error);
}

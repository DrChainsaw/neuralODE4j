package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for providing state of a solver
 *
 * @author Christian Skarby
 */
public interface SolverState {

    /**
     * Return the given stage of the derivative of the state
     * @param stage wanted stage of derivative
     * @return The derivative of the given stage
     */
    INDArray getStateDot(long stage);

    /**
     * Return the current state
     * @return the current state
     */
     INDArray getCurrentState();

    /**
     * Return the current time state
     * @return the current time state
     */
    INDArray time();

    /**
     * Return coefficients for calculating midpoints for the given solver
     * @return coefficients for calculating midpoints for the given solver
     */
    double[] getInterpolationMidpoints();
}

package ode.solve.api;

import ode.solve.impl.util.SolverState;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Listens to time steps taken by ODE solvers
 *
 * @author Christian Skarby
 */
public interface StepListener {

    /**
     * Indicates that solver has begun
     * @param t Desired time interval. t[0] is start time and t[1] is stop time.
     * @param y0 starting state (i.e. state at t[0])
     */
    void begin(INDArray t, INDArray y0);

    /**
     * Indicates a step has been taken
     * @param solverState Current state of the solver
     * @param step Taken step
     * @param error Estimated error
     */
    void step(SolverState solverState, INDArray step, INDArray error);

    /**
     * Indicates that previous step was the last step taken
     */
    void done();
}

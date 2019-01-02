package ode.solve.impl.listen;

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
     * @param currTime Current time (after step has been taken
     * @param step Taken step
     * @param error Estimated error
     * @param y current state (i.e estimated state at currTime)
     */
    void step(INDArray currTime, INDArray step, INDArray error, INDArray y);

    /**
     * Indicates that previous step was the last step taken
     */
    void done();
}

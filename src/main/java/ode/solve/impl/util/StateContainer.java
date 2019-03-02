package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Simple container for solver state.
 *
 * @author Christian Skarby
 */
public class StateContainer implements SolverState {

    private final INDArray currentTime;
    private final INDArray currentState;
    private final INDArray currentStateDot;

    public StateContainer(double currentTime, double[] currentState, double[] currentStateDot) {
        this(Nd4j.scalar(currentTime), Nd4j.create(currentState), Nd4j.create(currentStateDot));
    }

    public StateContainer(INDArray currentTime, INDArray currentState, INDArray currentStateDot) {
        this.currentTime = currentTime;
        this.currentState = currentState;
        this.currentStateDot = currentStateDot;
    }

    @Override
    public INDArray getStateDot(long stage) {
        return currentStateDot;
    }

    @Override
    public INDArray getCurrentState() {
        return currentState;
    }

    @Override
    public INDArray time() {
        return currentTime;
    }
}

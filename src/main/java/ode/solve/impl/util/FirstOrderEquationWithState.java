package ode.solve.impl.util;

import ode.solve.api.FirstOrderEquation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Combines a {@link FirstOrderEquation} with the state needed to evaluate it
 *
 * @author Christian Skarby
 */
public class FirstOrderEquationWithState {

    private final FirstOrderEquation equation;
    private final INDArray time;
    private final State state;

    private final static class State {
        private final INDArray currentState;
        private final INDArray workingState;
        private final INDArray stateDot;
        private final INDArray timeOffset;

        private State(INDArray currentState, long nrofStages) {
            this.currentState = currentState;
            this.workingState = currentState.dup();
            this.stateDot =  Nd4j.create(nrofStages, currentState.length());
            this.timeOffset = Nd4j.zeros(1);
        }

        void step(INDArray step, long stage) {
            timeOffset.assign(step);
            workingState.assign(currentState.add(stateDot.getRow(stage).mul(step)));
        }
    }

    public FirstOrderEquationWithState(
            FirstOrderEquation equation,
            INDArray time,
            INDArray state,
            long nrofStages) {
        if(!time.isScalar()) {
            throw new IllegalArgumentException("Expected time to be a scalar! Was: " + time);
        }
        this.equation = equation;
        this.time = time;
        this.state = new State(state, nrofStages);

    }

    /**
     * Calculates the derivative of the current working state for the current time
     * @param stage which stage to update
     */
    public void calculateDerivative(long stage) {
        equation.calculateDerivative(state.workingState, time.add(state.timeOffset), state.stateDot.getRow(stage));
    }

    public INDArray getFullStateDot() {
        return state.stateDot;
    }

    public INDArray getStateDot(long stage) {
        return state.stateDot.getRow(stage);
    }

    public INDArray getCurrentState() {
        return state.currentState;
    }

    /**
     * Updates the working state by taking the given step along the given stage of the derivative from the current state
     * @param step Step to take
     * @param stage Which stage to use to take the step
     */
    public void step(INDArray step, long stage) {
        state.step(step, stage);
    }

}

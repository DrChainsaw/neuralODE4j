package ode.solve.impl.util;

import ode.solve.api.FirstOrderEquation;
import ode.solve.impl.AdaptiveRungeKuttaSolver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
        private final INDArray y; // Last value of y. May be of any shape
        private final INDArray yWorking; // Working state to be updated by stepping through yDot. Same shape as y
        private final INDArray yDotK; // Matrix of flattened yDots, one row per stage in solver
        private final INDArray timeOffset; // Current time offset

        private State(INDArray y, long nrofStages) {
            this.y = y;
            this.yWorking = y.dup();
            this.yDotK = Nd4j.create(nrofStages, y.length());
            this.timeOffset = Nd4j.zeros(1);
        }

        void step(INDArray step, long stage) {
            timeOffset.assign(step);
            yWorking.assign(y.add(getStateDot(stage).mul(step)));
        }

        void stepAccum(INDArray stepCoeffPerStage, INDArray step, long startStage) {
            // yWorking = y + (stepCoeffPerStage*step) . yDot[0:startState, :]) where . is dot product
            timeOffset.assign(step);
            yWorking.assign(
                    y.add((stepCoeffPerStage.mul(step)).mmul(
                                    yDotK.get(NDArrayIndex.interval(0, startStage), NDArrayIndex.all())
                            )));
        }

        public INDArray getStateDot(long stage) {
            return yDotK.getRow(stage).reshape(y.shape());
        }
    }

    public FirstOrderEquationWithState(
            FirstOrderEquation equation,
            INDArray time,
            INDArray state,
            long nrofStages) {
        if (!time.isScalar()) {
            throw new IllegalArgumentException("Expected time to be a scalar! Was: " + time);
        }
        this.equation = equation;
        this.time = time;
        this.state = new State(state, nrofStages);

    }

    /**
     * Calculates the derivative of the current working state for the current time
     *
     * @param stage which stage to update
     */
    public void calculateDerivative(long stage) {
        equation.calculateDerivative(
                state.yWorking,
                time.add(state.timeOffset),
                state.getStateDot(stage));
    }

    public INDArray getStateDot(long stage) {
        return state.getStateDot(stage);
    }

    public INDArray getCurrentState() {
        return state.y;
    }

    public INDArray estimateError(AdaptiveRungeKuttaSolver.MseComputation mseComputation) {
        return mseComputation.estimateMse(state.yDotK, state.y, state.yWorking, state.timeOffset);
    }

    public INDArray time() {
        return time;
    }

    /**
     * Updates the working state by taking the given step along the given stage of the derivative from the current state
     *
     * @param step  Step to take
     * @param stage Which stage to use to take the step
     */
    public void step(INDArray step, long stage) {
        state.step(step, stage);
    }

    public void stepAccum(INDArray stepCoeffPerStage, INDArray step, long startStage) {
        state.stepAccum(stepCoeffPerStage, step, startStage);
    }
}

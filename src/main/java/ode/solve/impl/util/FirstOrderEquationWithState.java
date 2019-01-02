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

        void step(INDArray stepCoeffPerStage, INDArray step) {
            // yWorking = y + (stepCoeffPerStage*step) . yDot[0:startState, :]) where . is dot product
            timeOffset.assign(step);
            yWorking.assign(y);
                    yWorking.addi((stepCoeffPerStage.mul(step)).mmul(
                                    yDotK.get(NDArrayIndex.interval(0, stepCoeffPerStage.length()), NDArrayIndex.all())
                            ).reshape(y.shape()));
        }

        INDArray getStateDot(long stage) {
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
                state.getStateDot(stage)); // Note, stateDot of the given stage will be updated by this operation
    }

    /**
     * Return the given stage of the derivative of the state
     * @param stage wanted stage of derivative
     * @return The derivative of the given stage
     */
    public INDArray getStateDot(long stage) {
        return state.getStateDot(stage);
    }

    /**
     * Return the current state
     * @return the current state
     */
    public INDArray getCurrentState() {
        return state.y;
    }

    /**
     * Estimate the error using the given {@link AdaptiveRungeKuttaSolver.MseComputation}
     * @param mseComputation Strategy for computing the error
     * @return the computed error
     */
    public INDArray estimateError(AdaptiveRungeKuttaSolver.MseComputation mseComputation) {
        return mseComputation.estimateMse(state.yDotK, state.y, state.yWorking, state.timeOffset);
    }

    /**
     * Return the current time state
     * @return
     */
    public INDArray time() {
        return time;
    }

    /**
     * Update the working state by taking a step accumulated over all stages up the the given stage. The base step is
     * weighted with the given coefficients for each stage.
     * @param stepCoeffPerStage Weights for each stage (must be of shape [1, nrofStages]
     * @param step Base step. Must be a scalar
     */
    public void step(INDArray stepCoeffPerStage, INDArray step) {
        state.step(stepCoeffPerStage, step);
    }

    /**
     * Update the current state to the working state. This also includes shifting the derivative so that previous
     * stage 1 becomes new stage 0.
     */
    public void update() {
        time.addi(state.timeOffset);
        state.y.assign(state.yWorking);
        state.yDotK.putRow(0, state.yDotK.getRow(state.yDotK.rows()-1));
    }
}

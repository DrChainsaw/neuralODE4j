package ode.solve.impl.util;

import ode.solve.api.FirstOrderEquation;
import ode.solve.impl.AdaptiveRungeKuttaSolver;
import ode.solve.impl.DormandPrince54Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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

    public INDArray getFullStateDot() {
        return state.yDotK;
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




    public static void main(String[] args) {
        final INDArray yDotKIa = Nd4j.randn(new long[]{7, 4});
        final INDArray y0Ia = Nd4j.randn(new long[]{1, 4});
        final INDArray yTmpIa = y0Ia.dup();
        final INDArray[] aIa = DormandPrince54Solver.butcherTableu.a;
        final INDArray stepSizeIa = Nd4j.create(1).putScalar(0, 0.123);

        final double[][] yDotK = yDotKIa.toDoubleMatrix();
        final double[] y0 = y0Ia.toDoubleVector();
        final double[] yTmp = yTmpIa.toDoubleVector();
        final double[][] a = Stream.of(aIa).map(INDArray::toDoubleVector).collect(Collectors.toList()).toArray(new double[0][0]);
        final double stepSize = stepSizeIa.getDouble(0);

        for (int k = 1; k < 7; ++k) {
            for (int j = 0; j < y0.length; ++j) {
                double sum = a[k - 1][0] * yDotK[0][j];
                for (int l = 1; l < k; ++l) {
                    sum += a[k - 1][l] * yDotK[l][j];
                }
                yTmp[j] = y0[j] + stepSize * sum;
            }
            System.out.println("Ref1: " + Arrays.toString(yTmp));


            for (int j = 0; j < y0.length; ++j) {
                double sum = 0;
                for (int l = 0; l < k; ++l) {
                    sum += a[k - 1][l] * yDotK[l][j];
                }
                yTmp[j] = y0[j] + stepSize * sum;
            }
            System.out.println("Ref2: " + Arrays.toString(yTmp));

            final INDArray sum = Nd4j.zeros(y0Ia.shape());
            for (int l = 0; l < k; ++l) {
                sum.addi(yDotKIa.getRow(l).mul(aIa[k - 1].getScalar(l)));
            }
            yTmpIa.assign(y0Ia.add(sum.mul(stepSizeIa)));
            System.out.println("act1: " + Arrays.toString(yTmpIa.toDoubleVector()));

            yTmpIa.assign(y0Ia.add((aIa[k - 1].mul(stepSizeIa)).mmul(yDotKIa.get(NDArrayIndex.interval(0, k), NDArrayIndex.all()))));
            System.out.println("act2: " + Arrays.toString(yTmpIa.toDoubleVector()));


            System.out.println("*******************************************************************************");
        }
    }
}

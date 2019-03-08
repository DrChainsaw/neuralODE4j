package ode.solve.impl.util;

import ode.solve.api.StepListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.And;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.indexing.conditions.LessThan;

import java.util.Arrays;

/**
 * Samples {@link SolverState} at given time indexes using an {@link Interpolation}. Useful to extract multiple time
 * steps from a single time step solver. Advantage compared to calling the solver once per time step pair is fewer
 * function evaluations.
 *
 * @author Christian Skarby
 */
public class InterpolatingStepListener implements StepListener {

    private final INDArray wantedTimeInds;
    private final INDArray yInterpol;
    private final INDArrayIndex[] yInterpolAccess;
    private State state;

    // TODO: Get this from somewhere else, maybe SolverState
    private final static double[] DPS_C_MID = {
            6025192743d / 30085553152d / 2d, 0, 51252292925d / 65400821598d / 2d, -2691868925d / 45128329728d / 2d,
            187940372067d / 1594534317056d / 2d, -1776094331 / 19743644256d / 2d, 11237099 / 235043384d / 2d};

    private final class State {
        private Interpolation interpolation = new Interpolation();
        private INDArray y0;
        private INDArray t0;
    }

    /**
     * Create an {@link InterpolatingStepListener}. Output for the each wanted time will be assigned along dimension 0
     * of the provided yInterpol. In other words, output for wantedTimes[x] can be accessed through
     * yInterpol.get(NDArrayIndex.point(x), NDArrayIndex.all(), NDArrayIndex.all(), ...)
     *
     * @param wantedTimes Time samples for which output is desired.
     * @param yInterpol   Will contain output from the provided {@link SolverState} at the desired times.
     */
    public InterpolatingStepListener(INDArray wantedTimes, INDArray yInterpol) {
        if (wantedTimes.length() != yInterpol.size(0)) {
            throw new IllegalArgumentException("Must have one wanted time per element in dimension 0 of yInterpol! " +
                    "wantedTimes shape: " + Arrays.toString(wantedTimes.shape()) + ", yInterpol shape: " +
                    Arrays.toString(yInterpol.shape()));
        }

        this.wantedTimeInds = wantedTimes;
        this.yInterpol = yInterpol;

        this.yInterpolAccess = new INDArrayIndex[yInterpol.rank()];
        for (int dim = 0; dim < yInterpolAccess.length; dim++) {
            yInterpolAccess[dim] = NDArrayIndex.all();
        }
    }

    @Override
    public void begin(INDArray t, INDArray y0) {
        this.state = new State();
        this.state.t0 = t.getScalar(0);
        this.state.y0 = y0;

        // Edge case: The first wanted time index is the start time -> user wants the starting state to be added to output
        if (state.t0.equalsWithEps(wantedTimeInds.getScalar(0), 1e-10)) {
            yInterpolAccess[0] = NDArrayIndex.point(0);
            yInterpol.put(yInterpolAccess, state.y0);
        }
    }

    @Override
    public void step(SolverState solverState, INDArray step, INDArray error) {
        final INDArray greaterThanTime;
        final INDArray lessThanTime;
        if(step.getDouble(0) > 0) {
            greaterThanTime = state.t0;
            lessThanTime = solverState.time();
        } else {
            greaterThanTime = solverState.time();
            lessThanTime = state.t0;
        }

        final INDArray timeInds = wantedTimeInds.cond(
                new And(
                        new GreaterThan(greaterThanTime.getDouble(0)),
                        new LessThan(lessThanTime.getDouble(0)))
        );

        if (timeInds.sumNumber().doubleValue() > 0) {
            fitInterpolationCoeffs(solverState, step);
            doInterpolation(timeInds, solverState.time());
        }

        state.t0 = solverState.time().dup();
        state.y0 = solverState.getCurrentState().dup();
    }

    private void fitInterpolationCoeffs(SolverState solverState, INDArray step) {
        final INDArray[] yDotStages = new INDArray[DPS_C_MID.length];
        for (int i = 0; i < yDotStages.length; i++) {
            yDotStages[i] = solverState.getStateDot(i);
        }

        final INDArray state0;
        final INDArray state1;
        if(step.getDouble(0) > 0) {
            state0 = state.y0;
            state1 = solverState.getCurrentState();
        } else {
            state0 = solverState.getCurrentState();
            state1 = state.y0;
        }

        final INDArray yMid = state.y0.add(scaledDotProduct(
                Nd4j.createUninitialized(state0.shape()),
                DPS_C_MID,
                yDotStages,
                step));

        state.interpolation.fitCoeffs(
                state0,
                state1,
                yMid,
                yDotStages[0],
                yDotStages[yDotStages.length - 1],
                step);
    }

    private INDArray scaledDotProduct(INDArray output, double[] factors, INDArray[] inputs, INDArray scale) {
        output.assign((inputs[0].mul(factors[0])).mul(scale));
        for (int i = 1; i < inputs.length; i++) {
            output.addi(inputs[i].mul(factors[i]).mul(scale));
        }
        return output;
    }

    private void doInterpolation(INDArray timeInds, INDArray tNew) {
        final int startInd = timeInds.argMax().getInt(0);
        final int stopInd = startInd + timeInds.sumNumber().intValue();

        for (int i = startInd; i < stopInd; i++) {
            yInterpolAccess[0] = NDArrayIndex.point(i);

            final double t0;
            final double t1;
            final double tWanted = wantedTimeInds.getDouble(i);
            if(state.t0.getDouble(0) < wantedTimeInds.getDouble(i)) {
                t0 = state.t0.getDouble(0);
                t1 = tNew.getDouble(0);
            } else {
                t0 = tNew.getDouble(0);
                t1 = state.t0.getDouble(0);
            }

            yInterpol.put(yInterpolAccess,
                    state.interpolation.interpolate(t0, t1, tWanted));
        }
    }

    @Override
    public void done() {

        // Edge case: User wants last time step to be added to interpolation
        if (state.t0.equalsWithEps(wantedTimeInds.getScalar(wantedTimeInds.length() - 1), 1e-10)) {
            yInterpolAccess[0] = NDArrayIndex.point(wantedTimeInds.length() - 1);
            yInterpol.put(yInterpolAccess, state.y0);
        }
    }
}

package ode.solve.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderMultiStepSolver;
import ode.solve.api.FirstOrderSolver;
import ode.solve.api.StepListener;
import ode.solve.impl.util.InterpolatingStepListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import java.util.Arrays;

/**
 * {@link FirstOrderMultiStepSolver} which uses an {@link InterpolatingStepListener} to estimate intermediate time
 * steps. Will invoke the the wrapped {@link FirstOrderSolver} with the first and last time steps and return the
 * interpolated state (i.e not the output from the wrapped solver).
 */
public class InterpolatingMultiStepSolver implements FirstOrderMultiStepSolver {

    private final FirstOrderSolver solver;

    public InterpolatingMultiStepSolver(FirstOrderSolver solver) {
        this.solver = solver;
    }

    @Override
    public INDArray integrate(FirstOrderEquation equation, INDArray t, INDArray y0, INDArray yOut) {
        if(!t.isVector()) {
            throw new IllegalStateException("t must be a vector! Was shape " + Arrays.toString(t.shape()));
        }

        final INDArray wantedTimes = t.get(NDArrayIndex.interval(1, t.length())).reshape(getTimeShapeForLength(t, t.length()-1));

        final InterpolatingStepListener interpolation = new InterpolatingStepListener(wantedTimes, yOut);

        addListener(interpolation);

        // Note: skip first time index in order to comply with API description
        final INDArray tStartEnd = t.get(new SpecifiedIndex(0, t.length()-1)).reshape(getTimeShapeForLength(t, 2));
        solver.integrate(equation, tStartEnd, y0, y0);

        clearListeners(interpolation);

        // interpolation has made sure yOut contains estimated solutions at desired times
        return yOut;
    }

    private long[] getTimeShapeForLength(INDArray t, long length) {
        final long[] shape = t.shape().clone();
        for(int i = 0; i < shape.length; i++) {
            if(shape[i] != 1) {
                shape[i] = length;
            }
        }
        return shape;
    }

    @Override
    public void addListener(StepListener... listeners) {
        solver.addListener(listeners);
    }

    @Override
    public void clearListeners(StepListener... listeners) {
        solver.clearListeners(listeners);
    }
}

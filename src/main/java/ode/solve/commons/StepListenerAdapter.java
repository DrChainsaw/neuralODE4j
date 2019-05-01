package ode.solve.commons;

import ode.solve.api.StepListener;
import ode.solve.impl.util.StateContainer;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ode.sampling.StepHandler;
import org.apache.commons.math3.ode.sampling.StepInterpolator;
import org.nd4j.linalg.factory.Nd4j;

public class StepListenerAdapter implements StepHandler {

    private final StepListener wrappedListener;
    private long[] shape;

    public StepListenerAdapter(StepListener wrappedListener) {
        this.wrappedListener = wrappedListener;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }

    @Override
    public void init(double t0, double[] y0, double t) {
        wrappedListener.begin(Nd4j.create(new double[]{t0, t}), Nd4j.create(y0).reshape(shape));
    }

    @Override
    public void handleStep(StepInterpolator interpolator, boolean isLast) throws MaxCountExceededException {
        wrappedListener.step(
                new StateContainer(
                        Nd4j.scalar(interpolator.getCurrentTime()),
                        Nd4j.create(interpolator.getInterpolatedState()).reshape(shape),
                        Nd4j.create(interpolator.getInterpolatedDerivatives()).reshape(shape)),
                Nd4j.create(1).assign(interpolator.getCurrentTime() - interpolator.getPreviousTime()),
                null // error not observable :(
        );
    }
}

package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Calculates adaptive Runge-Kutta steps
 *
 * @author Christian Skarby
 */
public class AdaptiveRungeKuttaStepPolicy implements StepPolicy{

    private final static INDArray MIN_H = Nd4j.create(new double[]{1e-6});

    private final SolverConfig config;
    private final int order;

    public AdaptiveRungeKuttaStepPolicy(SolverConfig config, int order) {
        this.config = config;
        this.order = order;
    }

    @Override
    public INDArray initializeStep(FirstOrderEquationWithState equation, INDArray t) {

        equation.calculateDerivative(0);

        final INDArray scal = abs(equation.getCurrentState()).mul(config.getRelTol()).add(config.getAbsTol());
        final INDArray ratio = equation.getCurrentState().div(scal);
        final INDArray yOnScale2 = ratio.muli(ratio).sum();

        // Reuse ratio variable
        ratio.assign(equation.getStateDot(0).div(scal));
        final INDArray yDotOnScale2 = ratio.muli(ratio).sum();

        final INDArray h = ((yOnScale2.getDouble(0) < 1.0e-10) || (yDotOnScale2.getDouble(0) < 1.0e-10)) ?
                MIN_H : sqrt(yOnScale2.divi(yDotOnScale2)).muli(0.01);

        final boolean backward = t.argMax().getInt(0) == 0;
        if (backward) {
            h.negi();
        }

        equation.step(h, 0);
        equation.calculateDerivative(1);

        // Reuse ratio variable
        ratio.assign(equation.getStateDot(1).sub(equation.getStateDot(0)).divi(scal));
        ratio.muli(ratio);
        final INDArray yDDotOnScale = sqrt(ratio.sum()).divi(h);

        // step size is computed such that
        // h^order * max (||y'/tol||, ||y''/tol||) = 0.01
        final INDArray maxInv2 = max(sqrt(yDotOnScale2), yDDotOnScale);
        final INDArray h1 = maxInv2.getDouble(0) < 1e-15 ?
                max(MIN_H, abs(h).muli(0.001)) :
                pow(maxInv2.rdivi(0.01), 1d / order);

        h.assign(min(abs(h).muli(100), h1));
        h.assign(max(h, abs(t.getColumn(0)).muli(1e-12)));

        h.assign(max(config.getMinStep(), h));
        h.assign(min(config.getMaxStep(), h));

        if (backward) {
            h.negi();
        }

        return h;
    }

}

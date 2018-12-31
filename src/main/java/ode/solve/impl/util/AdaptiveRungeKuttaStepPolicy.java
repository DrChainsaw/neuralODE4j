package ode.solve.impl.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * Calculates adaptive Runge-Kutta steps
 *
 * @author Christian Skarby
 */
public class AdaptiveRungeKuttaStepPolicy implements StepPolicy {

    private final static INDArray MIN_H = Nd4j.create(1).putScalar(0, 1e-6);

    private final SolverConfig config;
    private final StepConfig stepConfig;
    private final double exp;

    public static class StepConfig {
        private final INDArray maxGrowth;
        private final INDArray minReduction;
        private final INDArray safety;
        private final int order;

        public static Builder builder(int order) {
            return new Builder(order);
        }

        public StepConfig(final double maxGrowth,
                          final double minReduction,
                          final double safety,
                          final int order) {
            this.maxGrowth = Nd4j.create(1).assign(maxGrowth);
            this.minReduction = Nd4j.create(1).assign(minReduction);
            this.safety = Nd4j.create(1).assign(safety);
            this.order = order;
        }
        
        public static class Builder {

            private double maxGrowth = 10;
            private double minReduction = 0.2;
            private double safety = 0.9;
            private final int order;

            private Builder(int order) {
                this.order = order;
            }

            /** Set the maximal growth factor for stepsize control.
             * @param maxGrowth maximal growth factor
             */
            public Builder setMaxGrowth(double maxGrowth) {
                this.maxGrowth = maxGrowth; return this;
            }

            /** Set the minimal reduction factor for stepsize control.
             * @param minReduction minimal reduction factor
             */
            public Builder setMinReduction(double minReduction) {
                this.minReduction = minReduction; return this;
            }

            /** Set the safety factor for stepsize control.
             * @param safety safety factor
             */
            public Builder setSafety(double safety) {
                this.safety = safety; return this;
            }

            public StepConfig build() {
                return new StepConfig(maxGrowth, minReduction, safety, order);
            }
        }
    }

    public AdaptiveRungeKuttaStepPolicy(SolverConfig config, int order) {
        this(config, StepConfig.builder(order).build());
    }

    public AdaptiveRungeKuttaStepPolicy(SolverConfig config, StepConfig stepConfig) {
        this.config = config;
        this.stepConfig = stepConfig;
        this.exp = -1.0 / stepConfig.order;
    }

    @Override
    public INDArray initializeStep(FirstOrderEquationWithState equation, INDArray t) {

        equation.calculateDerivative(0);

        final INDArray scal = abs(equation.getCurrentState()).mul(config.getRelTol()).add(config.getAbsTol());
        INDArray ratio = equation.getCurrentState().div(scal);
        final INDArray yOnScale2 = ratio.muli(ratio).sum();

        // Reuse ratio variable
        ratio.assign(equation.getStateDot(0));
        ratio.divi(scal);
        final INDArray yDotOnScale2 = ratio.muli(ratio).sum();

        final INDArray step = ((yOnScale2.getDouble(0) < 1.0e-10) || (yDotOnScale2.getDouble(0) < 1.0e-10)) ?
                MIN_H : sqrt(yOnScale2.divi(yDotOnScale2)).muli(0.01);

        final boolean backward = t.argMax().getInt(0) == 0;
        if (backward) {
            step.negi();
        }

        equation.step(Nd4j.ones(1), step);
        equation.calculateDerivative(1);

        // Reuse ratio variable
        ratio.assign(equation.getStateDot(1));
        ratio.subi(equation.getStateDot(0)).divi(scal);
        ratio.muli(ratio);
        final INDArray yDDotOnScale = sqrt(ratio.sum()).divi(step);

        // step size is computed such that
        // step^order * max (||y'/tol||, ||y''/tol||) = 0.01
        final INDArray maxInv2 = max(sqrt(yDotOnScale2), yDDotOnScale);
        final INDArray step1 = maxInv2.getDouble(0) < 1e-15 ?
                max(MIN_H, abs(step).muli(0.001)) :
                pow(maxInv2.rdivi(0.01), 1d / stepConfig.order);

        step.assign(min(abs(step).muli(100), step1));
        step.assign(max(step, abs(t.getColumn(0)).muli(1e-12)));

        step.assign(max(config.getMinStep(), step));
        step.assign(min(config.getMaxStep(), step));

        if (backward) {
            step.negi();
        }

        return step;
    }

    @Override
    public INDArray step(INDArray step, INDArray error) {
        final INDArray sign = sign(step);
        return bound(stepFactor(error).muli(step).muli(sign), config.getMaxStep(), config.getMinStep()).muli(sign);
    }

    private INDArray stepFactor(INDArray error) {
        return bound(stepConfig.safety.mul(pow(error, exp)), stepConfig.maxGrowth, stepConfig.minReduction);
    }

    private static INDArray bound(INDArray var, INDArray upper, INDArray lower) {
        return min(upper, max(lower, var));
    }
}

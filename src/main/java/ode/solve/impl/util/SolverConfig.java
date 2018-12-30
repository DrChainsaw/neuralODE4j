package ode.solve.impl.util;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Configuration parameters for {@link ode.solve.api.FirstOrderSolver}s
 */
@Getter
public class SolverConfig {

    private final INDArray absTol;
    private final INDArray relTol;
    private final INDArray minStep;
    private final INDArray maxStep;

    public SolverConfig(
            double absoluteTolerance,
            double relativeTolerance,
            double minStep,
            double maxStep) {
        if(minStep >= maxStep) {
            throw new IllegalArgumentException("Max step smaller than min step! Swapped arguments? max: " + maxStep + " min " + minStep );
        }

        this.absTol = Nd4j.create(1).assign(absoluteTolerance);
        this.relTol = Nd4j.create(1).assign(relativeTolerance);
        this.minStep = Nd4j.create(1).assign(minStep);
        this.maxStep = Nd4j.create(1).assign(maxStep);
    }

}

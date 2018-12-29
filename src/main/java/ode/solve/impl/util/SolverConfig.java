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
        this.absTol = Nd4j.create(1).assign(absoluteTolerance);
        this.relTol = Nd4j.create(1).assign(relativeTolerance);
        this.minStep = Nd4j.create(1).assign(minStep);
        this.maxStep = Nd4j.create(1).assign(maxStep);
    }

}

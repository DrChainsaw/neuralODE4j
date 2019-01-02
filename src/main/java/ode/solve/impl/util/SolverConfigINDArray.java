package ode.solve.impl.util;

import lombok.Getter;
import ode.solve.conf.SolverConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Solver configuration expressed with {@link INDArray}s
 *
 * @author Christian Skarby
 */
@Getter
public class SolverConfigINDArray {

    private final INDArray absTol;
    private final INDArray relTol;
    private final INDArray minStep;
    private final INDArray maxStep;

    public SolverConfigINDArray(SolverConfig config) {
        this(
                config.getAbsTol(),
                config.getRelTol(),
                config.getMinStep(),
                config.getMaxStep());
    }

    public SolverConfigINDArray(
            double absoluteTolerance,
            double relativeTolerance,
            double minStep,
            double maxStep) {
        if (minStep >= maxStep) {
            throw new IllegalArgumentException("Max step smaller than min step! Swapped arguments? max: " + maxStep + " min " + minStep);
        }
        this.absTol = Nd4j.scalar(absoluteTolerance);
        this.relTol = Nd4j.scalar(relativeTolerance);
        this.minStep = Nd4j.scalar(minStep);
        this.maxStep = Nd4j.scalar(maxStep);
    }
}

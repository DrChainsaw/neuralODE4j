package ode.solve.conf;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Serializable configuration parameters for {@link ode.solve.api.FirstOrderSolver}s
 */
@Data
public class SolverConfig {

    private final double absTol;
    private final double relTol;
    private final double minStep;
    private final double maxStep;

    public SolverConfig(
            @JsonProperty("absoluteTolerance") double absoluteTolerance,
            @JsonProperty("relativeTolerance") double relativeTolerance,
            @JsonProperty("minStep") double minStep,
            @JsonProperty("maxStep") double maxStep) {
        if(minStep >= maxStep) {
            throw new IllegalArgumentException("Max step smaller than min step! Swapped arguments? max: " + maxStep + " min " + minStep );
        }
        this.absTol = absoluteTolerance;
        this.relTol = relativeTolerance;
        this.minStep = minStep;
        this.maxStep = maxStep;
    }

}

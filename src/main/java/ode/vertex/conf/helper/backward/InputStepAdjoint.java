package ode.vertex.conf.helper.backward;

import lombok.Data;
import ode.solve.api.FirstOrderSolverConf;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Serializable configuration of an {@link ode.vertex.impl.helper.backward.InputStepAdjoint}
 *
 * @author Christian Skarby
 */
@Data
public class InputStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolverConf solverConf;
    private final int timeInputIndex;

    public InputStepAdjoint(
            @JsonProperty("solverConf") FirstOrderSolverConf solverConf,
            @JsonProperty("timeInputIndex") int timeInputIndex) {
        this.solverConf = solverConf;
        this.timeInputIndex = timeInputIndex;
    }

    @Override
    public ode.vertex.impl.helper.backward.OdeHelperBackward instantiate() {
        return new ode.vertex.impl.helper.backward.InputStepAdjoint(solverConf.instantiate(), timeInputIndex);
    }

    @Override
    public InputStepAdjoint clone() {
        return new InputStepAdjoint(solverConf.clone(), timeInputIndex);
    }
}

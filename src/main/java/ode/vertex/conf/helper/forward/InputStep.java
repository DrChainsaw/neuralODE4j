package ode.vertex.conf.helper.forward;

import lombok.Data;
import ode.solve.api.FirstOrderSolverConf;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Serializable configuration of an {@link ode.vertex.impl.helper.forward.InputStep}
 *
 * @author Christian Skarby
 */
@Data
public class InputStep implements OdeHelperForward {

    private final FirstOrderSolverConf solverConf;
    private final int timeInputIndex;

    public InputStep(
            @JsonProperty("solverConf") FirstOrderSolverConf solverConf,
            @JsonProperty("timeInputIndex") int timeInputIndex) {
        this.solverConf = solverConf;
        this.timeInputIndex = timeInputIndex;
    }

    @Override
    public ode.vertex.impl.helper.forward.OdeHelperForward instantiate() {
        return new ode.vertex.impl.helper.forward.InputStep(solverConf.instantiate(), timeInputIndex);
    }
}

package ode.vertex.conf.helper.forward;

import lombok.Data;
import ode.solve.api.FirstOrderSolverConf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.jackson.shaded.NDArrayDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArraySerializer;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Serializable configuration for {@link ode.vertex.impl.helper.forward.FixedStep}
 */
@Data
public class FixedStep implements OdeHelperForward{

    private final FirstOrderSolverConf solverConf;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private final INDArray time;

    public FixedStep(
            @JsonProperty("solverConf") FirstOrderSolverConf solverConf,
            @JsonProperty("time") INDArray time) {
        this.solverConf = solverConf;
        this.time = time;
    }

    @Override
    public ode.vertex.impl.helper.forward.OdeHelperForward instantiate() {
        return new ode.vertex.impl.helper.forward.FixedStep(solverConf.instantiate(), time);
    }
}

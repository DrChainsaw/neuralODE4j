package ode.vertex.conf.helper.backward;

import lombok.Data;
import ode.solve.api.FirstOrderSolverConf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.jackson.shaded.NDArrayDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArraySerializer;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Serializable configuration for {@link ode.vertex.impl.helper.backward.BackpropagateAdjoint}
 *
 * @author Christian Skarby
 */
@Data
public class FixedStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolverConf solverConf;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private final INDArray time;

    public FixedStepAdjoint(
            @JsonProperty("solverConf") FirstOrderSolverConf solverConf,
            @JsonProperty("time") INDArray time) {
        this.solverConf = solverConf;
        this.time = time;
    }

    @Override
    public ode.vertex.impl.helper.backward.OdeHelperBackward instantiate() {
        return new ode.vertex.impl.helper.backward.FixedStepAdjoint(solverConf.instantiate(), time);
    }

    @Override
    public FixedStepAdjoint clone() {
        return new FixedStepAdjoint(solverConf.clone(), time.dup());
    }
}

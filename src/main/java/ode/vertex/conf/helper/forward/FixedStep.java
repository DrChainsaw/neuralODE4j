package ode.vertex.conf.helper.forward;

import lombok.Data;
import ode.solve.api.FirstOrderSolverConf;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.jackson.shaded.NDArrayDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArraySerializer;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Serializable configuration for {@link ode.vertex.impl.helper.forward.FixedStep}
 *
 * @author Christian Skarby
 */
@Data
public class FixedStep implements OdeHelperForward{

    private final FirstOrderSolverConf solverConf;
    @JsonSerialize(using = NDArraySerializer.class)
    @JsonDeserialize(using = NDArrayDeSerializer.class)
    private final INDArray time;
    private final boolean interpolateIfMultiStep;

    public FixedStep(
            @JsonProperty("solverConf") FirstOrderSolverConf solverConf,
            @JsonProperty("time") INDArray time,
            @JsonProperty("interpolateIfMultiStep") boolean interpolateIfMultiStep) {
        this.solverConf = solverConf;
        this.time = time;
        this.interpolateIfMultiStep = interpolateIfMultiStep;
    }

    @Override
    public ode.vertex.impl.helper.forward.OdeHelperForward instantiate() {
        return new ode.vertex.impl.helper.forward.FixedStep(solverConf.instantiate(), time, interpolateIfMultiStep);
    }

    @Override
    public int nrofTimeInputs() {
        return 0;
    }

    @Override
    public FixedStep clone() {
        return new FixedStep(solverConf.clone(), time.dup(), interpolateIfMultiStep);
    }

    @Override
    public InputType getOutputType(ComputationGraphConfiguration conf, InputType... vertexInputs) throws InvalidInputTypeException {

        final OutputTypeHelper confHelper = new OutputTypeFromConfig();
        if(time.length() == 2) {
            return confHelper.getOutputType(conf, vertexInputs);
        }

        final InputType[] withTime = new InputType[vertexInputs.length+1];
        System.arraycopy(vertexInputs, 0, withTime, 0, vertexInputs.length);
        withTime[vertexInputs.length] =  InputType.inferInputType(this.time);
        return new OutputTypeAddTimeAsDimension(vertexInputs.length, confHelper).getOutputType(conf, withTime);
    }
}

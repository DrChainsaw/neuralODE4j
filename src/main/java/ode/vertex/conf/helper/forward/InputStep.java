package ode.vertex.conf.helper.forward;

import lombok.Data;
import ode.solve.api.FirstOrderSolverConf;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Serializable configuration of an {@link ode.vertex.impl.helper.forward.InputStep}
 *
 * @author Christian Skarby
 */
@Data
public class InputStep implements OdeHelperForward {

    private final FirstOrderSolverConf solverConf;
    private final int timeInputIndex;
    private final boolean interpolateIfMultiStep;

    public InputStep(
            @JsonProperty("solverConf") FirstOrderSolverConf solverConf,
            @JsonProperty("timeInputIndex") int timeInputIndex,
            @JsonProperty("interpolateIfMultiStep") boolean interpolateIfMultiStep) {
        this.solverConf = solverConf;
        this.timeInputIndex = timeInputIndex;
        this.interpolateIfMultiStep = interpolateIfMultiStep;
    }

    @Override
    public ode.vertex.impl.helper.forward.OdeHelperForward instantiate() {
        return new ode.vertex.impl.helper.forward.InputStep(solverConf.instantiate(), timeInputIndex, interpolateIfMultiStep);
    }

    @Override
    public int nrofTimeInputs() {
        return 1;
    }

    @Override
    public InputStep clone() {
        return new InputStep(solverConf.clone(), timeInputIndex, interpolateIfMultiStep);
    }

    @Override
    public InputType getOutputType(ComputationGraphConfiguration conf, InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length <= timeInputIndex) {
            throw new InvalidInputTypeException("Time input index was not part of input types!!");
        }

        final InputType timeInput = vertexInputs[timeInputIndex];
        if(timeInput.arrayElementsPerExample() > 2) {
            return new OutputTypeAddTimeAsDimension(timeInputIndex, new OutputTypeFromConfig()).getOutputType(conf, vertexInputs);
        }
        List<InputType> inputTypeList = new ArrayList<>();
        for (int i = 0; i < vertexInputs.length; i++) {
            if (i != timeInputIndex) {
                inputTypeList.add(vertexInputs[i]);
            }
        }
        return new OutputTypeFromConfig().getOutputType(conf, inputTypeList.toArray(new InputType[0]));
    }
}

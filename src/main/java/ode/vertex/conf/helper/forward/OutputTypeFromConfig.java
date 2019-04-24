package ode.vertex.conf.helper.forward;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;

import java.util.List;
import java.util.Map;

/**
 * Determines output type from the given configuration
 *
 * @author Christian Skarby
 */
public class OutputTypeFromConfig implements OutputTypeHelper {

    @Override
    public InputType getOutputType(ComputationGraphConfiguration conf, InputType... vertexInputs) throws InvalidInputTypeException {
        final Map<String, InputType> inputTypeMap = conf.getLayerActivationTypes(vertexInputs);

        for(List<String> vertexInputList: conf.getVertexInputs().values()) {
            inputTypeMap.keySet().removeAll(vertexInputList);
        }

        if(inputTypeMap.size() != 1) {
            throw new IllegalStateException("Can only support one single output! Make sure the computation graph has " +
                    "exactly one layer which is not input to any other layer. Vertices " + inputTypeMap.keySet() + " are " +
                    "not input to any other layer!");
        }

        return inputTypeMap.values().iterator().next();
    }
}

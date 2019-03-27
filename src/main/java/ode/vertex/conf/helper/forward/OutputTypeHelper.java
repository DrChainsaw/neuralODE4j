package ode.vertex.conf.helper.forward;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;

/**
 * Interface for determining the shape of output given shape of inputs.
 *
 * @author Christian Skarby
 */
public interface OutputTypeHelper {

    /**
     * Return the format of the input for the given InputTypes
     * @param conf {@link ComputationGraphConfiguration} which will be used as the derivative
     * @param vertexInputs Inputs to the vertex
     * @return an {@link InputType} for the next vertex in the graph
     * @throws InvalidInputTypeException
     */
    InputType getOutputType(ComputationGraphConfiguration conf, InputType... vertexInputs) throws InvalidInputTypeException;
}

package ode.vertex.conf.helper;

import ode.vertex.impl.helper.GraphInputOutput;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Factory for {@link GraphInputOutput}
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface GraphInputOutputFactory {

    /**
     * Create a new {@link GraphInputOutput} for the given input
     *
     * @param input input, typically activations from previous vertices
     * @return a new {@link GraphInputOutput}
     */
    GraphInputOutput create(INDArray[] input);

    GraphInputOutputFactory clone();

    /**
     * Returns the input types to the graph. Typically a matter of either adding one extra element for the time or not.
     * @param vertexInputs Inputs to vertex
     * @return InputTypes to use in graph
     */
    InputType[] getInputType(InputType ... vertexInputs);
}

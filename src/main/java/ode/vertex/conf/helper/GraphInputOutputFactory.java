package ode.vertex.conf.helper;

import ode.vertex.impl.helper.GraphInputOutput;
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
}

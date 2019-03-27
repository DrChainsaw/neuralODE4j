package ode.vertex.impl.gradview.parname;

import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * A mapping between parameter and vertex names to a combined name. Also capable of reverse mapping.
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface ParamNameMapping {

    /**
     * Map layerName and paramName to a new parameter name which is unique for the given input
     * @param vertexName Name of layer
     * @param paramName Name of parameter
     * @return A combined namn
     */
    String map(String vertexName, String paramName);

    /**
     * Reverse mapping. In other words, {@code mapping.reverseMap(mapping.map(vertexName, paramName)); } returns
     * {@code [vertexName, paramName]}
     * @param combinedName Combined name to reverese map
     * @return a Pair where vertexName is the first member and paramName is the second.
     */
    Pair<String, String> reverseMap(String combinedName);

}

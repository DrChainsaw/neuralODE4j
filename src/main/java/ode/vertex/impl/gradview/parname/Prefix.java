package ode.vertex.impl.gradview.parname;

import lombok.Data;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * {@link ParamNameMapping} which adds a prefix to another mapping
 *
 * @author Christian Skarby
 */
@Data
public class Prefix implements ParamNameMapping {

    private final ParamNameMapping mapping;
    private final String prefix;

    public Prefix(@JsonProperty("prefix") String prefix,
                  @JsonProperty("mapping") ParamNameMapping mapping) {
        this.mapping = mapping;
        this.prefix = prefix;
    }

    @Override
    public String map(String vertexName, String paramName) {
        return prefix + mapping.map(vertexName, paramName);
    }

    @Override
    public Pair<String, String> reverseMap(String combinedName) {
        return mapping.reverseMap(combinedName.substring(prefix.length()));
    }
}

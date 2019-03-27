package ode.vertex.impl.gradview.parname;

import lombok.Data;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * {@link ParamNameMapping} which concatenates the names
 *
 * @author Christian Skarby
 */
@Data
public class Concat implements ParamNameMapping {

    private final String concatStr;

    public Concat() {
        this("-");
    }

    public Concat(@JsonProperty("concatStr") String concatStr) {
        this.concatStr = concatStr;
    }


    @Override
    public String map(String vertexName, String paramName) {
        return vertexName + concatStr + paramName;
    }

    @Override
    public Pair<String, String> reverseMap(String combinedName) {
        final String[] split = combinedName.split(concatStr);
        if(split.length != 2) {
            throw new IllegalArgumentException("Can not reverse mapping for " + combinedName);
        }
        return new Pair<>(split[0], split[1]);
    }
}

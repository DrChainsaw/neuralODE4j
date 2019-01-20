package ode.vertex.conf.helper.backward;

import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Serializable configuration of an {@link ode.vertex.impl.helper.backward.OdeHelperBackward}
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface OdeHelperBackward {

    /**
     * Instantiate the helper
     * @return a New {@link ode.vertex.impl.helper.backward.OdeHelperBackward}
     */
    ode.vertex.impl.helper.backward.OdeHelperBackward instantiate();
}

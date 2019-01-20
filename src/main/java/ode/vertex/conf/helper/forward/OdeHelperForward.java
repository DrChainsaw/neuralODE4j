package ode.vertex.conf.helper.forward;

import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Serializable configuration of an {@link ode.vertex.impl.helper.forward.OdeHelperForward}
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface OdeHelperForward {

    /**
     * Instantiate the helper
     * @return a New {@link ode.vertex.impl.helper.forward.OdeHelperForward}
     */
    ode.vertex.impl.helper.forward.OdeHelperForward instantiate();
}

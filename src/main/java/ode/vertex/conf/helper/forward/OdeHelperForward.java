package ode.vertex.conf.helper.forward;

import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Serializable configuration of an {@link ode.vertex.impl.helper.forward.OdeHelperForward}
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface OdeHelperForward extends OutputTypeHelper {

    /**
     * Instantiate the helper
     * @return a New {@link ode.vertex.impl.helper.forward.OdeHelperForward}
     */
    ode.vertex.impl.helper.forward.OdeHelperForward instantiate();

    /**
     * How many time inputs are needed
     * @return the number of needed time inputs
     */
    int nrofTimeInputs();


    /**
     * Clone the configuration
     * @return a clone of the configuration
     */
    OdeHelperForward clone();
}

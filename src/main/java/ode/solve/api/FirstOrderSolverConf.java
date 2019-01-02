package ode.solve.api;

import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Base interface for configuration of a {@link FirstOrderSolver}. Main motivation is to enable (de)serialization without
 * forcing actual implementations to carry alot of extra weight. Basically same approach as for layers/vertices in other
 * dl4j models.
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface FirstOrderSolverConf extends Cloneable {

    FirstOrderSolver instantiate();

    /**
     * Clone the config
     * @return a clone
     */
    FirstOrderSolverConf clone();

}

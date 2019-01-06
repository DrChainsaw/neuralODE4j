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


    // TODO: Don't really like that listeners can be added to conf as it creates ambiguity as to whether they will be
    //  serialized (they wont). However, when using solvers as a part of an OdeVertex it is not straightforward to get
    //  access to the solver instance
    /**
     * Add {@link StepListener}s which will be notified of steps taken
     * @param listeners listeners to add
     */
    void addListeners(StepListener... listeners);

    /**
     * Clear the given listeners. Clear all listeners if empty
     * @param listeners listeners to remove
     */
    void clearListeners(StepListener ... listeners);

}

package ode.vertex.impl.helper.backward;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Maps output from back propagation from a {@link org.deeplearning4j.nn.graph.ComputationGraph} to an
 * {@link AugmentedDynamics}. Main use case is for specifying whether current time is input or not.
 */
public interface GraphBackwardsOutput  {

    /**
     * Updates the given {@link AugmentedDynamics} with the result of back propagation
     * @param gradients Result of back propagation
     * @param augmentedDynamics State to update
     */
    void update(List<INDArray> gradients, AugmentedDynamics augmentedDynamics);

}

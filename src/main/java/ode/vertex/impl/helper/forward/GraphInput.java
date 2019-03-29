package ode.vertex.impl.helper.forward;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Produces input to a {@link org.deeplearning4j.nn.graph.ComputationGraph} when used as an ODE. Main use case is
 * for specifying whether current time is input or not.
 *
 * @author Christian Skarby
 */
public interface GraphInput {

    /**
     * Return array of inputs from a (possibly concatenated) state y and a current time t
     * @param y Current state
     * @param t Current time
     * @return Input arrays
     */
    INDArray[] getInputsFrom(INDArray y, INDArray t);

    /**
     * Initial value for forward pass
     * @return The initial value for forward pass
     */
    INDArray y0();

    /**
     * Removes input for given index
     * @param index Input index to remove
     * @return GraphInputOutput with input index removed and the removed input array
     */
    Pair<? extends GraphInput, INDArray> removeInput(int index);

}

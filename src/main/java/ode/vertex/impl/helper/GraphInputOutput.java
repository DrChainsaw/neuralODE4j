package ode.vertex.impl.helper;

import ode.vertex.impl.helper.backward.GraphBackwardsOutput;
import ode.vertex.impl.helper.forward.GraphInput;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Union of {@link GraphInput} and {@link GraphBackwardsOutput}
 *
 * @author Christian Skarby
 */
public interface GraphInputOutput extends GraphInput, GraphBackwardsOutput {

    /**
     * Removes input for given index
     * @param index Input index to remove
     * @return GraphInputOutput with input index removed and the removed input array
     */
    Pair<GraphInputOutput, INDArray> removeInput(int index);

}

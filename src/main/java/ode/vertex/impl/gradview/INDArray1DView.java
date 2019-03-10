package ode.vertex.impl.gradview;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *  A 1D view of one or more {@link INDArray}s.
 *
 * @author Christian Skarby
 */
public interface INDArray1DView {

    /**
     * Sets the view to the given {@link INDArray}
     * @param toAssign view will be set to this. Must be same size as view
     */
    void assignFrom(INDArray toAssign);

    /**
     * Sets the values of the given {@link INDArray} to the values of the view
     * @param assignTo will be set to state of the view. Must be same size as view
     */
    void assignTo(INDArray assignTo);

    /**
     * Return the current length (total number of elements) of the view
     * @return the current length of the view
     */
    long length();
}

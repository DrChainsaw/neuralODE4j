package ode.vertex.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A non-contiguous view of a 1D INDArray
 *
 * @author Christian Skarby
 */
public class NonContiguous1DView {

    private final List<INDArray> view = new ArrayList<>();
    private long length = 0;

    public void addView(INDArray array, long begin, long end) {
        INDArray viewSlice = array.get(NDArrayIndex.interval(begin, end));
        addView(viewSlice);
    }

    public void addView(INDArray viewSlice) {
        if(viewSlice.isColumnVectorOrScalar()) {
            throw new IllegalArgumentException("Must be vector or scalar! Had shape: " + Arrays.toString(viewSlice.shape()));
        }
        length += viewSlice.length();
        view.add(viewSlice);
    }

    /**
     * Sets the view to the given {@link INDArray}
     * @param toAssign view will be set to this. Must be same size as view
     */
    public void assignFrom(INDArray toAssign) {
        if(toAssign.length() != length) {
            throw new IllegalArgumentException("Array to assignFrom must have same length! " +
                    "This length: " + length +" array length: " + toAssign.length());
        }

        if(toAssign.rank() != 1) {
            throw new IllegalArgumentException("Array toAssign must have rank 1!");
        }

        long ptr = 0;
        for(INDArray viewSlice: view) {
            viewSlice.assign(toAssign.get(NDArrayIndex.interval(ptr, ptr + viewSlice.length())));
            ptr += viewSlice.length();
        }
    }

    /**
     * Sets the values of the given {@link INDArray} to the values of the view
     * @param assignTo will be set to state of the view. Must be same size as view
     */
    public void assignTo(INDArray assignTo) {
        if(assignTo.length() != length) {
            throw new IllegalArgumentException("Array assignTo must have same length! " +
                    "This length: " + length +" array length: " + assignTo.length());
        }
        if(assignTo.rank() != 1) {
            throw new IllegalArgumentException("Array assignTo must have rank 1!");
        }

        long ptr = 0;
        for(INDArray viewSlice: view) {
            assignTo.put(new INDArrayIndex[] {NDArrayIndex.interval(ptr, ptr + viewSlice.length())}, viewSlice);
            ptr += viewSlice.length();
        }
    }

    /**
     * Return the current length (total number of elements) of the view
     * @return the current length of the view
     */
    public long length() {
        return length;
    }

    /**
     * Clears the view
     */
    public void clear() {
        view.clear();
        length = 0;
    }
}

package ode.vertex.impl.gradview;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * A non-contiguous view of a 1D INDArray
 *
 * @author Christian Skarby
 */
public class NonContiguous1DView implements INDArray1DView {

    private final List<INDArray> view = new ArrayList<>();
    private long length = 0;

    public void addView(INDArray viewSlice) {
        length += viewSlice.length();
        view.add(viewSlice);
    }

    @Override
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
            viewSlice.assign(toAssign.get(NDArrayIndex.interval(ptr, ptr + viewSlice.length())).reshape(viewSlice.shape()));
            ptr += viewSlice.length();
        }
    }

    @Override
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
            assignTo.put(new INDArrayIndex[] {NDArrayIndex.interval(ptr, ptr + viewSlice.length())}, viewSlice.reshape(viewSlice.length()));
            ptr += viewSlice.length();
        }
    }

    @Override
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

    @Override
    public String toString() {
        return view.toString();
    }
}

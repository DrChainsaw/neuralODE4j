package ode.vertex.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;

/**
 * Utility class for accumulating {@link INDArray}s
 *
 * @author Christian Skarby
 */
class NDArrayIndexAccumulator {

    private final INDArrayIndex[] state;
    private final INDArray array;

    NDArrayIndexAccumulator(INDArray array) {
        this.array = array;
        state = new INDArrayIndex[array.shape().length];
        for(int i = 0; i < array.shape().length; i++) {
            state[i] = new NDArrayIndexAll(true);
        }
    }

    NDArrayIndexAccumulator increment(INDArray toAdd) {
        for(int dim = 0; dim < toAdd.shape().length; dim++) {
            if(toAdd.size(dim) != array.size(dim)) {
                final long curr = state[dim] instanceof NDArrayIndexAll ? 0 : state[dim].end();
                state[dim] = NDArrayIndex.interval(curr, curr + toAdd.size(dim));
            }
        }
        array.put(state, toAdd);
        return this;
    }
}

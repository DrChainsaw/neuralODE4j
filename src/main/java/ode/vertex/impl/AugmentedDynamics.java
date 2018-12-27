package ode.vertex.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

/**
 * Augmented Dynamics used for adjoint back propagation method from https://arxiv.org/pdf/1806.07366.pdf
 *
 * @author Christian Skarby
 */
class AugmentedDynamics {

    private final INDArray z;
    private final INDArray zAdjoint;
    private final INDArray paramAdjoint;
    private final INDArray tAdjoint;


    AugmentedDynamics(INDArray zAug, long[] zShape, long[] paramShape, long[] tShape) {
        this(
                zAug.get(NDArrayIndex.interval(0, length(zShape))).reshape(zShape),
                zAug.get(NDArrayIndex.interval(length(zShape), 2 * length(zShape))).reshape(zShape),
                zAug.get(NDArrayIndex.interval(2 * length(zShape), 2 * length(zShape) + length(paramShape))).reshape(paramShape),
                zAug.get(NDArrayIndex.interval(2 * length(zShape) + length(paramShape), 2 * length(zShape) + length(paramShape) + length(tShape))).reshape(tShape));
    }

    AugmentedDynamics(INDArray z, INDArray zAdjoint, INDArray paramAdjoint, INDArray tAdjoint) {
        this.z = z;
        this.zAdjoint = zAdjoint;
        this.paramAdjoint = paramAdjoint;
        this.tAdjoint = tAdjoint;
    }

    private static long length(long[] shape) {
        long length = 1;
        for(long dimElems: shape) {
            length *= dimElems;
        }
        return length;
    }

    void updateFrom(INDArray zAug) {
        long offset = updateSubsetArr(z, zAug, 0);
        offset = updateSubsetArr(zAdjoint, zAug, offset);
        offset = updateSubsetArr(paramAdjoint, zAug, offset);
        updateSubsetArr(tAdjoint, zAug, offset);
    }

    void transferTo(INDArray zAug) {
        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(Nd4j.toFlattened(z))
                .increment(Nd4j.toFlattened(zAdjoint))
                .increment(Nd4j.toFlattened(paramAdjoint))
                .increment(Nd4j.toFlattened(tAdjoint));
    }

    private static long updateSubsetArr(INDArray subsetArr, INDArray arr, long offset) {
        subsetArr.reshape(1, subsetArr.length()).assign(arr.get(NDArrayIndex.interval(offset, subsetArr.length() + offset)));
        return offset + subsetArr.length();
    }

    void updateZAdjoint(final List<INDArray> epsilons) {
        long lastInd = 0;
        for (int i = 0; i < epsilons.size(); i++) {
            final INDArray eps = epsilons.get(i);
            zAdjoint.reshape(1 ,zAdjoint.length()).put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(lastInd, eps.length())}, Nd4j.toFlattened(eps));
            lastInd += eps.length();
        }
    }

    void updateParamAdjoint(INDArray gradient) {
        paramAdjoint.assign(gradient);
    }

    public INDArray getZ() {
        return z;
    }

    public INDArray getZAdjoint() {
        return zAdjoint;
    }

    public INDArray getParamAdjoint() {
        return paramAdjoint;
    }
}

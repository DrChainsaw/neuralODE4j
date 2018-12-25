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

    private final INDArray epsilon;
    private INDArray[] lastEpsilons;

    AugmentedDynamics(double[] zAug, long nrofZ, long nrofParam, long nrofT, long[] epsShape) {
        this(Nd4j.create(zAug), nrofZ, nrofT, nrofParam, epsShape);
    }

    AugmentedDynamics(INDArray zAug, long nrofZ, long nrofParam, long nrofT, long[] epsShape) {
        this(
                zAug.get(NDArrayIndex.interval(0, nrofZ)),
                zAug.get(NDArrayIndex.interval(nrofZ, 2 * nrofZ)).reshape(epsShape),
                zAug.get(NDArrayIndex.interval(2 * nrofZ, 2 * nrofZ + nrofParam)),
                zAug.get(NDArrayIndex.interval(2 * nrofZ + nrofParam, 2 * nrofZ + nrofParam + nrofT)));
    }

    AugmentedDynamics(INDArray z, INDArray zAdjoint, INDArray paramAdjoint, INDArray tAdjoint) {
        this.z = z;
        this.epsilon = zAdjoint;
        this.zAdjoint = zAdjoint.reshape(1, zAdjoint.length());
        this.paramAdjoint = paramAdjoint;
        this.tAdjoint = tAdjoint;
    }

    long getNrofElements() {
        return z.length() + zAdjoint.length() + paramAdjoint.length() + tAdjoint.length();
    }

    void updateFrom(INDArray zAug) {
        long offset = updateSubsetArr(z, zAug, 0);
        offset = updateSubsetArr(zAdjoint, zAug, offset);
        offset = updateSubsetArr(paramAdjoint, zAug, offset);
        updateSubsetArr(tAdjoint, zAug, offset);
    }

    void transferTo(INDArray zAug) {
        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(z)
                .increment(zAdjoint)
                .increment(paramAdjoint)
                .increment(tAdjoint);
    }

    private static long updateSubsetArr(INDArray subsetArr, INDArray arr, long offset) {
        subsetArr.assign(arr.get(NDArrayIndex.interval(offset, subsetArr.length() + offset)));
        return offset + subsetArr.length();
    }


    void update(double[] zAug) {
        int offset = updateArr(zAug, z, 0);
        offset = updateArr(zAug, zAdjoint, offset);
        offset = updateArr(zAug, paramAdjoint, offset);
        updateArr(zAug, tAdjoint, offset);
    }

    void transferTo(double[] zAug) {
        int offset = updateVec(zAug, z, 0);
        offset = updateVec(zAug, zAdjoint, offset);
        offset = updateVec(zAug, paramAdjoint, offset);
        updateArr(zAug, tAdjoint, offset);
    }

    private static int updateArr(double[] vec, INDArray arr, int offset) {
        for (int i = 0; i < arr.length(); i++) {
            arr.putScalar(i, vec[i + offset]);
        }
        return offset + (int) arr.length();
    }

    private static int updateVec(double[] vec, INDArray arr, int offset) {
        for (int i = 0; i < arr.length(); i++) {
            vec[i + offset] = arr.getDouble(i);
        }
        return offset + (int) arr.length();
    }

    void updateZAdjoint(final List<INDArray> epsilons) {
        long lastInd = 0;
        for (INDArray eps : epsilons) {
            // Note: This is a view of epsilon so this will update epsilon too
            zAdjoint.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(lastInd, eps.length())}, Nd4j.toFlattened(eps));
            lastInd += eps.length();
        }
        lastEpsilons = epsilons.toArray(new INDArray[0]);
    }

    void updateParamAdjoint(INDArray gradient) {
        paramAdjoint.assign(gradient);
    }

    INDArray[] getLastEpsilons() {
        return lastEpsilons;
    }

    public INDArray getEpsilon() {
        return epsilon;
    }

    public INDArray getZ() {
        return z;
    }
}

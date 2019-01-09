package ode.vertex.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

/**
 * Augmented Dynamics used for adjoint back propagation method from https://arxiv.org/pdf/1806.07366.pdf
 *
 * @author Christian Skarby
 */
class AugmentedDynamics {

    private final INDArray augStateFlat;
    private final INDArray z;
    private final INDArray zAdjoint;
    private final INDArray paramAdjoint;
    private final INDArray tAdjoint;


    AugmentedDynamics(INDArray zAug, long[] zShape, long[] paramShape, long[] tShape) {
        this(
                zAug,
                zAug.get(NDArrayIndex.interval(0, length(zShape))).reshape(zShape),
                zAug.get(NDArrayIndex.interval(length(zShape), 2 * length(zShape))).reshape(zShape),
                zAug.get(NDArrayIndex.interval(2 * length(zShape), 2 * length(zShape) + length(paramShape))).reshape(paramShape),
                zAug.get(NDArrayIndex.interval(2 * length(zShape) + length(paramShape), 2 * length(zShape) + length(paramShape) + length(tShape))).reshape(tShape));
    }

    AugmentedDynamics(INDArray augStateFlat, INDArray z, INDArray zAdjoint, INDArray paramAdjoint, INDArray tAdjoint) {
        this.augStateFlat = augStateFlat;
        this.z = z;
        this.zAdjoint = zAdjoint;
        this.paramAdjoint = paramAdjoint;
        this.tAdjoint = tAdjoint;
    }

    private static long length(long[] shape) {
        long length = 1;
        for (long dimElems : shape) {
            length *= dimElems;
        }
        return length;
    }

    void updateFrom(INDArray zAug) {
        augStateFlat.assign(zAug);
    }

    void transferTo(INDArray zAug) {
        zAug.assign(augStateFlat);
    }

    void updateZAdjoint(final List<INDArray> epsilons) {
        long lastInd = 0;
        for (int i = 0; i < epsilons.size(); i++) {
            final INDArray eps = epsilons.get(i);
            zAdjoint.reshape(1, zAdjoint.length())
                    .put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(lastInd, eps.length())},
                            eps.reshape(new long[]{1, eps.length()}));
            lastInd += eps.length();
        }
    }

    public INDArray z() {
        return z;
    }

    public INDArray zAdjoint() {
        return zAdjoint;
    }

    public INDArray paramAdjoint() {
        return paramAdjoint;
    }

    public INDArray tAdjoint() {
        return tAdjoint;
    }
}

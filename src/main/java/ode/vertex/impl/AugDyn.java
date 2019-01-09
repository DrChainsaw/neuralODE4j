package ode.vertex.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

public class AugDyn {

    private INDArray augStateFlat;
    private final Views views;


    private static class Views {
        private final ParView z;
        private final ParView zAdjoint;
        private final ParView paramAdjoint;
        private final ParView tAdjoint;

        private Views(long[] zShape, long[] paramShape, long[] tShape) {
            this.z = new ParView(NDArrayIndex.interval(0, length(zShape)), zShape);
            this.zAdjoint = new ParView(NDArrayIndex.interval(length(zShape), 2 * length(zShape)), zShape);
            this.paramAdjoint = new ParView(NDArrayIndex.interval(2 * length(zShape), 2 * length(zShape) + length(paramShape)), paramShape);
            this.tAdjoint = new ParView(NDArrayIndex.interval(
                    2 * length(zShape) + length(paramShape),
                    2 * length(zShape) + length(paramShape) + length(tShape)), tShape);
        }

        private static long length(long[] shape) {
            long length = 1;
            for (long dimElems : shape) {
                length *= dimElems;
            }
            return length;
        }

    }

    private static class ParView {

        private INDArrayIndex indices;
        private final long[] shape;


        private ParView(INDArrayIndex indices, long[] shape) {
            this.indices = indices;
            this.shape = shape;
        }

        private INDArray getView(INDArray flatState) {
            indices.reset();
            return flatState.get(indices).reshape(shape);
        }
    }

    AugDyn(INDArray zAug, long[] zShape, long[] paramShape, long[] tShape) {
        this.augStateFlat = zAug;
        this.views = new Views(zShape, paramShape, tShape);
    }

    void setWorkingState(INDArray zAug) {
        augStateFlat = zAug;
    }

    void updateZAdjoint(final List<INDArray> epsilons) {
        long lastInd = 0;
        final INDArray zAdjoint = views.zAdjoint.getView(augStateFlat);
        for (int i = 0; i < epsilons.size(); i++) {
            final INDArray eps = epsilons.get(i);
            zAdjoint.reshape(1, zAdjoint.length())
                    .put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(lastInd, eps.length())},
                            eps.reshape(new long[]{1, eps.length()}));
            lastInd += eps.length();
        }
    }

    public INDArray z() {
        return views.z.getView(augStateFlat);
    }

    public INDArray zAdjoint() {
        return views.zAdjoint.getView(augStateFlat);
    }

    public INDArray paramAdjoint() {
        return views.paramAdjoint.getView(augStateFlat);
    }

    public INDArray tAdjoint() {
        return views.tAdjoint.getView(augStateFlat);
    }

}

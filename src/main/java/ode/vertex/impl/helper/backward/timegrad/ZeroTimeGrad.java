package ode.vertex.impl.helper.backward.timegrad;

import ode.solve.api.FirstOrderEquation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * {@link TimeGrad} which does not compute and time gradients, but returns a dummy gradient so that array sizes
 * are correct
 *
 * @author Christian Skarby
 */
public class ZeroTimeGrad implements TimeGrad {

    private final int timeIndex;

    /**
     * Factory for this class
     */
    public static class Factory implements TimeGrad.Factory {

        private final int timeIndex;

        public Factory(int timeIndex) {
            this.timeIndex = timeIndex;
        }

        @Override
        public TimeGrad create() {
            return new ZeroTimeGrad(timeIndex);
        }
    }


    public ZeroTimeGrad(int timeIndex) {
        this.timeIndex = timeIndex;
    }

    @Override
    public INDArray calcTimeAdjointT1(FirstOrderEquation equation, INDArray zt1, INDArray time) {
        return Nd4j.empty();
    }

    @Override
    public INDArray[] createLossGradient(INDArray zAdjoint, INDArray tAdjoint) {
        final INDArray[] epsilons = new INDArray[2];
        for (int i = 0; i < epsilons.length; i++) {
            if (i != timeIndex) {
                epsilons[i] = zAdjoint;
            }
        }
        epsilons[timeIndex] = Nd4j.zeros(1,2);
        return epsilons;
    }
}

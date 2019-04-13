package ode.vertex.impl.helper.backward.timegrad;

import ode.solve.api.FirstOrderEquation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Calculates a time gradient for a single time step.
 *
 * @author Christian Skarby
 */
public class CalcTimeGrad implements TimeGrad {

    private final INDArray dL_dzt1_time;
    private final INDArray startTime;
    private final int timeIndex;

    private INDArray dL_dt1;

    /**
     * Factory for this class
     */
    public static class Factory implements TimeGrad.Factory {

        private final INDArray dL_dzt1_time;
        private final INDArray startTime;
        private final int timeIndex;

        public Factory(INDArray dL_dzt1_time, int timeIndex) {
            this(dL_dzt1_time, Nd4j.scalar(0), timeIndex);
        }

        public Factory(INDArray dL_dzt1_time, INDArray startTime, int timeIndex) {
            this.dL_dzt1_time = dL_dzt1_time;
            this.startTime = startTime;
            this.timeIndex = timeIndex;
        }

        @Override
        public TimeGrad create() {
            return new CalcTimeGrad(dL_dzt1_time, startTime, timeIndex);
        }
    }

    public CalcTimeGrad(INDArray dL_dzt1_time, INDArray startTime, int timeIndex) {
        this.dL_dzt1_time = dL_dzt1_time;
        this.startTime = startTime;
        this.timeIndex = timeIndex;
    }

    @Override
    public INDArray calcTimeAdjointT1(FirstOrderEquation equation, INDArray zt1, INDArray time) {
        final INDArray dzt1_dt1 = equation.calculateDerivative(zt1, time.getColumn(1), zt1.dup());

        this.dL_dt1 = dL_dzt1_time.reshape(1, dzt1_dt1.length())
                .mmul(dzt1_dt1.reshape(dzt1_dt1.length(), 1));
        return startTime.subi(dL_dt1);
    }

    @Override
    public INDArray[] createLossGradient(INDArray zAdjoint, INDArray tAdjoint) {
        if(dL_dt1 == null) {
            throw new IllegalStateException("Must compute dL / dt1 before creating loss gradient!");
        }

        final INDArray[] epsilons = new INDArray[2];
        for (int i = 0; i < epsilons.length; i++) {
            if (i != timeIndex) {
                epsilons[i] = zAdjoint;
            }
        }
        epsilons[timeIndex] = Nd4j.hstack(tAdjoint, dL_dt1);
        return epsilons;
    }
}

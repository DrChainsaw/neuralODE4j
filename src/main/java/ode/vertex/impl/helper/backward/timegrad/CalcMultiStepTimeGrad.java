package ode.vertex.impl.helper.backward.timegrad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Calculates a time gradient for multiple time steps.
 *
 * @author Christian Skarby
 */
public class CalcMultiStepTimeGrad implements MultiStepTimeGrad {

    private final INDArray timeGradient;
    private final int timeIndex;

    private double lastTime = 0;

    /**
     * Factory for this class.
     */
    public static class Factory implements MultiStepTimeGrad.Factory {

        private final INDArray time;
        private final int timeIndex;

        public Factory(INDArray time, int timeIndex) {
            this.time = time;
            this.timeIndex = timeIndex;
        }

        @Override
        public MultiStepTimeGrad create() {
            return new CalcMultiStepTimeGrad(time, timeIndex);
        }
    }

    public CalcMultiStepTimeGrad(INDArray time, int timeIndex) {
        timeGradient = Nd4j.zeros(time.shape());
        this.timeIndex = timeIndex;
    }

    @Override
    public void updateStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        timeGradient.put(timeIndexer, gradients[timeIndex]);
        lastTime -= timeGradient.get(timeIndexer).getDouble(1);
    }

    @Override
    public void updateLastStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        timeIndexer[1] = NDArrayIndex.point(0);
        timeGradient.put(timeIndexer, lastTime);
        gradients[timeIndex] = timeGradient;
    }

    @Override
    public TimeGrad.Factory createSingleStepFactory(INDArray dL_dzt1_time) {
        return new CalcTimeGrad.Factory(dL_dzt1_time, timeIndex);
    }
}

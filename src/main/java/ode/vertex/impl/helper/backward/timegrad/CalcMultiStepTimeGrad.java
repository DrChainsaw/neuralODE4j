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

    private final INDArray lastTime = Nd4j.scalar(0);

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
    public void prepareStep(INDArray[] lastGradients, INDArray dL_dzt) {
        if(lastGradients == null) return;

        dL_dzt.addi(lastGradients[(timeIndex+1) % lastGradients.length]);
    }

    @Override
    public void updateStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        timeGradient.put(timeIndexer, gradients[timeIndex]);
    }

    @Override
    public INDArray[] updateLastStep(INDArrayIndex[] timeIndexer, INDArray[] gradients, INDArray dL_dzt0) {
        timeIndexer[1] = NDArrayIndex.point(0);
        timeGradient.put(timeIndexer, lastTime);
        gradients[timeIndex] = timeGradient;
        gradients[(timeIndex + 1) % gradients.length].addi(dL_dzt0);
        return gradients;
    }

    @Override
    public TimeGrad.Factory createSingleStepFactory(INDArray dL_dzt1_time) {
        return new CalcTimeGrad.Factory(dL_dzt1_time, lastTime, timeIndex);
    }
}

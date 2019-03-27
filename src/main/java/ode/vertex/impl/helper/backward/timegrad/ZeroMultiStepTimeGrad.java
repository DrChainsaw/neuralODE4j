package ode.vertex.impl.helper.backward.timegrad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

/**
 * {@link MultiStepTimeGrad} which does not compute and time gradients, but returns a dummy gradient so that array sizes
 * are correct
 *
 * @author Christian Skarby
 */
public class ZeroMultiStepTimeGrad implements MultiStepTimeGrad {

    private final int timeIndex;
    private final INDArray time;

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
            return new ZeroMultiStepTimeGrad(time, timeIndex);
        }
    }

    public ZeroMultiStepTimeGrad(INDArray time, int timeIndex) {
        this.time = time;
        this.timeIndex = timeIndex;
    }

    @Override
    public void prepareStep(INDArray[] lastGradients, INDArray dL_dzt) {
        if(lastGradients == null) return;

        dL_dzt.addi(lastGradients[(timeIndex+1) % lastGradients.length]);
    }

    @Override
    public void updateStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        // Do nothing
    }

    @Override
    public INDArray[] updateLastStep(INDArrayIndex[] timeIndexer, INDArray[] gradients, INDArray dL_dzt) {

        final INDArray[] toRet = new INDArray[2];
        toRet[timeIndex] = Nd4j.zerosLike(time);
        final int notTimeIndex = (timeIndex + 1) % toRet.length;
        toRet[notTimeIndex] = gradients[notTimeIndex % gradients.length];
        gradients[notTimeIndex % gradients.length].addi(dL_dzt);

        return toRet;
    }

    @Override
    public TimeGrad.Factory createSingleStepFactory(INDArray dL_dzt1_time) {
        return NoTimeGrad.factory;
    }
}

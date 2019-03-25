package ode.vertex.impl.helper.backward.timegrad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;

/**
 * Does not calculcate any time gradients
 *
 * @author Christian Skarby
 */
public class NoMultiStepTimeGrad implements MultiStepTimeGrad {

    // Singleton because stateless
    private final static MultiStepTimeGrad noTimeGrad = new NoMultiStepTimeGrad();

    /**
     * Factory for this class
     */
    public static MultiStepTimeGrad.Factory factory = new MultiStepTimeGrad.Factory() {

        @Override
        public MultiStepTimeGrad create() {
            return noTimeGrad;
        }
    };

    @Override
    public void prepareStep(INDArray[] lastGradients, INDArray dL_dzt) {
        if(lastGradients == null) return;

        dL_dzt.addi(lastGradients[0]);
    }

    @Override
    public void updateStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        // Do nothing
    }

    @Override
    public INDArray[] updateLastStep(INDArrayIndex[] timeIndexer, INDArray[] gradients, INDArray dL_dzt) {
        gradients[0].addi(dL_dzt);
        return gradients;
    }

    @Override
    public TimeGrad.Factory createSingleStepFactory(INDArray dL_dzt1_time) {
        return NoTimeGrad.factory;
    }
}

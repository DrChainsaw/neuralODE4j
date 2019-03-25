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
    public void updateStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        // Do nothing
    }

    @Override
    public void updateLastStep(INDArrayIndex[] timeIndexer, INDArray[] gradients) {
        // Do nothing
    }

    @Override
    public TimeGrad.Factory createSingleStepFactory(INDArray dL_dzt1_time) {
        return NoTimeGrad.factory;
    }
}

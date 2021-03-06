package ode.vertex.impl.helper.backward.timegrad;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;

/**
 * Interface for handling time gradient w.r.t loss when multiple time steps are used. Main use case is to be able to
 * not calculate time gradients when not needed.
 *
 * @author Christian Skarby
 */
public interface MultiStepTimeGrad {

    /**
     * Factory for MultiStepTimeGrads
     */
    interface Factory {
        /**
         * Return a {@link MultiStepTimeGrad} instance
         * @return a {@link MultiStepTimeGrad} instance
         */
        MultiStepTimeGrad create();
    }

    /**
     * Update dL_dzt based on last steps gradients
     * @param lastGradients gradients from last time step
     * @param dL_dzt Loss gradient for z(t) for current time step
     */
    void prepareStep(INDArray[] lastGradients, INDArray dL_dzt);

    /**
     * Update after a single step backwards
     * @param timeIndexer Points to the current time step
     * @param gradients Current gradients
     */
    void updateStep(INDArrayIndex[] timeIndexer, INDArray[] gradients);

    /**
     * Update after the last step backwards has been performed. Note that this will update the input gradient
     * @param timeIndexer Points to the current time step
     * @param gradients Current gradients. Might be updated as a result
     * @param dL_dzt0 Loss gradient for z(t0).
     * @return Updated gradients
     */
    INDArray[] updateLastStep(INDArrayIndex[] timeIndexer, INDArray[] gradients, INDArray dL_dzt0);

    /**
     * Create an appropriate {@link TimeGrad.Factory}
     * @param dL_dzt1_time Loss gradient for calculation of time gradient
     * @return a {@link TimeGrad.Factory}
     */
    TimeGrad.Factory createSingleStepFactory(INDArray dL_dzt1_time);

}

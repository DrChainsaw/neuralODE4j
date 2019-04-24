package ode.vertex.impl.helper.backward.timegrad;

import ode.solve.api.FirstOrderEquation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for handling time gradient w.r.t loss for a single time step. Main use case is to be able to
 * not calculate time gradients when not needed.
 *
 * @author Christian Skarby
 */
public interface TimeGrad {

    /**
     * Factory for TimeGrads
     */
    interface Factory {
        /**
         * Return a {@link TimeGrad} instance
         * @return a {@link TimeGrad} instance
         */
        TimeGrad create();
    }

    /**
     * Calculate adjoint time for the last time point (t1)
     * @param equation Calculates dz(t1) / d(t1)
     * @param zt1 Value of z(t1)
     * @param time
     * @return adjoint time or empty if no time gradient needed
     */
    INDArray calcTimeAdjointT1(FirstOrderEquation equation, INDArray zt1, INDArray time);

    /**
     * Create loss gradient (a.k.a epsilons in dl4j) from adjoint state.
     * @param zAdjoint dL / dz(t0)
     * @param tAdjoint dL / dt0
     * @return Array of required loss gradients
     */
    INDArray[] createLossGradient(INDArray zAdjoint, INDArray tAdjoint);

}

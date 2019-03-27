package ode.vertex.conf.helper;

import ode.vertex.conf.helper.backward.OdeHelperBackward;
import ode.vertex.conf.helper.forward.OdeHelperForward;

/**
 * Convenience configuration of both {@link OdeHelperForward} and {@link OdeHelperBackward}.
 *
 * @author Christian Skarby
 */
public interface OdeHelper {

    /**
     * Create the helper config in the forward direction
     *
     * @return helper in forward direction
     */
    OdeHelperForward forward();

    /**
     * Create the helper config in the backward direction
     *
     * @return helper in backward direction
     */
    OdeHelperBackward backward();

}

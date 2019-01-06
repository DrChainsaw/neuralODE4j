package ode.solve.conf;

import ode.solve.api.FirstOrderEquation;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Simple equation (y = e^t) for testing
 *
 * @author Christian Skarby
 */
class ProbeEquation implements FirstOrderEquation {

    private int nrofCalls = 0;

    @Override
    public INDArray calculateDerivative(INDArray y, INDArray t, INDArray fy) {
        nrofCalls++;
        return fy;
    }

    void assertNrofCalls(int expected) {
        assertEquals("Incorect number of calls!", expected, nrofCalls);
    }

    void assertWasCalled() {
        assertTrue("Was not called!", 0 < nrofCalls);
    }
}

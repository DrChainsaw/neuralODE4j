package util.listen.step;

import ode.solve.api.StepListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test class for checking that listeners are called
 *
 * @author Christian Skarby
 */
public class ProbeStepListener implements StepListener {

    private int nrofBegin;
    private int nrofStep;
    private int nrofDone;

    @Override
    public void begin(INDArray t, INDArray y0) {
        nrofBegin++;
    }

    @Override
    public void step(INDArray currTime, INDArray step, INDArray error, INDArray y) {
        nrofStep++;
    }

    @Override
    public void done() {
        nrofDone++;
    }

    /**
     * Assert that the right number of calls has been done to each method
     *
     * @param expectedBegin Expected number of calls to begin
     * @param expectedStep  Expected number of calls to step
     * @param expectedDone  Expected number of calls to done
     */
    public void assertNrofCalls(int expectedBegin, int expectedStep, int expectedDone) {
        assertEquals("Incorrect number of calls!", expectedBegin, nrofBegin);
        assertEquals("Incorrect number of calls!", expectedStep, nrofStep);
        assertEquals("Incorrect number of calls!", expectedDone, nrofDone);
    }

    public void assertWasCalled() {
        assertTrue("Begin was not called!", 0 < nrofBegin);
        assertTrue("Step was not called!", 0 < nrofStep);
        assertTrue("Done was not called!", 0 < nrofDone);
    }
}

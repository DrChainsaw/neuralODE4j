package util.listen.step;

import ode.solve.api.StepListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link Mask}
 *
 * @author Christian Skarby
 */
public class MaskTest {

    /**
     * Test masking in forward direction
     */
    @Test
    public void forward() {
        final ProbeStepListener probeStepListener = new ProbeStepListener();
        final StepListener maskForward = Mask.forward(probeStepListener);
        testMask(probeStepListener, maskForward, Nd4j.linspace(-1, 1, 2), Nd4j.linspace(1, 0, 2));
    }

    /**
     * Test masking in backward direction
     */
    @Test
    public void backward() {
        final ProbeStepListener probeStepListener = new ProbeStepListener();
        final StepListener maskForward = Mask.backward(probeStepListener);
        testMask(probeStepListener, maskForward, Nd4j.linspace(10, 0, 2), Nd4j.linspace(-2, 3, 2));
    }

    private void testMask(ProbeStepListener probeStepListener, StepListener mask, INDArray masked, INDArray notMasked) {
        mask.begin(masked, Nd4j.zeros(0));
        mask.step(Nd4j.zeros(0), Nd4j.zeros(0), Nd4j.zeros(0), Nd4j.zeros(0));
        mask.done();

        probeStepListener.assertNrofCalls(0,0,0);

        mask.begin(notMasked, Nd4j.zeros(0));
        mask.step(Nd4j.zeros(0), Nd4j.zeros(0), Nd4j.zeros(0), Nd4j.zeros(0));
        mask.done();

        probeStepListener.assertNrofCalls(1,1,1);
    }


}
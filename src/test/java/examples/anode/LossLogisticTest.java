package examples.anode;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link LossLogistic}
 *
 * @author Christian Skarby
 */
public class LossLogisticTest {

    /**
     * Test that gradient and score are as expected when label and output are equal
     */
    @Test
    public void computeGradientAndScoreNoError() {
        final INDArray lab = Nd4j.ones(1);
        final INDArray out = lab.dup();
        Pair<Double, INDArray> gradAndScore = new LossLogistic()
                .computeGradientAndScore(lab, out, new ActivationIdentity(), null, true);

        assertEquals("Incorrect score!", 0.4519, gradAndScore.getFirst(), 1e-3);
        assertEquals("Incorrect gradient!", -0.3880, gradAndScore.getSecond().maxNumber().doubleValue(), 1e-3);
    }

    /**
     * Test the name
     */
    @Test
    public void name() {
        assertEquals("Incorrect name", "LossLogistic()", new LossLogistic().name());
    }
}
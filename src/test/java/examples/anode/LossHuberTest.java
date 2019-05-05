package examples.anode;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link LossHuber}
 *
 * @author Christian Skarby
 */
public class LossHuberTest {

    /**
     * Test that gradient and score make sense
     */
    @Test
    public void computeGradientAndScore() {
        final INDArray pred = Nd4j.create(new double[][]{{1, 10}, {-4, -2}});
        final INDArray label = Nd4j.create(new double[][]{{0.3, -2}, {18, -1.2}});

        final Pair<Double, INDArray> gradAndScore = new LossHuber().computeGradientAndScore(label, pred, new ActivationIdentity(), null, true);
        assertEquals("Incorrect score!", 8.3912, gradAndScore.getFirst(), 1e-3);
        assertEquals("Incorrect gradient!", Nd4j.create(new double[][]{{0.7, 1}, {-1, -0.8}}), gradAndScore.getSecond());
    }

    /**
     * Check the name
     */
    @Test
    public void name() {
        assertEquals("Incorrect name!", "LossHuber()", new LossHuber().name());
    }
}
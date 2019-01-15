package examples.spiral;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link NormElboLoss}
 *
 * @author Christian Skarby
 */
public class NormElboLossTest {

    /**
     * Test that score is according to expectation. Result was calculated from original repo.
     */
    @Test
    public void computeScore() {
        final double sigma = 0.3;
        final double mean = 2 * Math.PI;
        final ActivationListener qMean = new ActivationListener("mean");
        final ActivationListener qLogVar = new ActivationListener("logVar");
        final NormElboLoss loss = new NormElboLoss(mean, sigma, qMean, qLogVar);

        qMean.onForwardPass(null, Collections.singletonMap("mean", Nd4j.scalar(Math.log(mean))));
        qLogVar.onForwardPass(null, Collections.singletonMap("logVar", Nd4j.scalar(Math.log(sigma) * 2)));

        final INDArray output = Nd4j.create(new double[][]{
                {-1, 2, -3, 4, -5},
        });

        assertEquals("Incorrect loss!",1.0126971006393433,
                loss.computeScore(output, output, new ActivationIdentity(), null, false), 1e-4);
    }

    /**
     * Test that score array has correct dimension when a spiral is used as input
     */
    @Test
    public void computeScoreArraySpiral() {
        final double sigma = 0.3;
        final double mean = 2 * Math.PI;
        final ActivationListener qMean = new ActivationListener("mean");
        final ActivationListener qLogVar = new ActivationListener("logVar");
        final NormElboLoss loss = new NormElboLoss(mean, sigma, qMean, qLogVar);

        final int batchSize = 10;
        qMean.onForwardPass(null, Collections.singletonMap("mean", Nd4j.arange(batchSize*4).reshape(10, 4)));
        qLogVar.onForwardPass(null, Collections.singletonMap("logVar", Nd4j.arange(batchSize*4).reshape(10, 4)));

        final SpiralIterator.Generator gen = new SpiralIterator.Generator(
                new SpiralFactory(0, 0.3, 0, 2*Math.PI, 200),
                0.3, 50, new Random(666));
        final INDArray resultAndOutput = gen.generate(batchSize).getFeatures(0);

        final INDArray score = loss.computeScoreArray(resultAndOutput, resultAndOutput, new ActivationIdentity(), null);

        assertEquals("Incorrect size!", batchSize, score.length());

    }
}
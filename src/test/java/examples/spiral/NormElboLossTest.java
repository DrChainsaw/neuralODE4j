package examples.spiral;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Triple;

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
        final NormElboLoss loss = new NormElboLoss(mean, sigma, result -> new Triple<>(
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 5)),
                result.get(NDArrayIndex.all(), NDArrayIndex.point(5)),
                result.get(NDArrayIndex.all(), NDArrayIndex.point(6))));

        final INDArray labels = Nd4j.create(new double[][]{
                {-1, 2, -3, 4, -5},
        });

        final INDArray resultConc = Nd4j.concat(1, labels, Nd4j.scalar(Math.log(mean)), Nd4j.scalar(2 * Math.log(sigma)));

        assertEquals("Incorrect loss!",1.0126971006393433,
                loss.computeScore(labels, resultConc, new ActivationIdentity(), null, false), 1e-4);
    }

    /**
     * Test that score array has correct dimension when a spiral is used as input
     */
    @Test
    public void computeScoreArraySpiral() {
        final double sigma = 0.3;
        final double mean = 2 * Math.PI;
        final long nrofSamples = 50;
        final long nrofLatentDims = 4;
        final int batchSize = 10;

        final NormElboLoss loss = new NormElboLoss(mean, sigma, result -> new Triple<>(
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2*nrofSamples)).reshape(batchSize, nrofSamples, 2),
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(2*nrofSamples, 2*nrofSamples + nrofLatentDims)),
                result.get(NDArrayIndex.all(), NDArrayIndex.interval(2*nrofSamples+nrofLatentDims, 2*nrofSamples + 2*nrofLatentDims))
        ));



        final SpiralIterator.Generator gen = new SpiralIterator.Generator(
                new SpiralFactory(0, 0.3, 0, 2*Math.PI, 200),
                0.3, nrofSamples, new Random(666));
        final INDArray labels = gen.generate(batchSize).getFeatures(0);

        final INDArray resultConc = Nd4j.hstack(
                labels.reshape(batchSize, labels.length() / batchSize),
                Nd4j.arange(batchSize*nrofLatentDims).reshape(batchSize, nrofLatentDims),
                Nd4j.arange(batchSize*nrofLatentDims).reshape(batchSize, nrofLatentDims));


        final INDArray score = loss.computeScoreArray(labels, resultConc, new ActivationIdentity(), null);

        assertEquals("Incorrect size!", batchSize, score.length());

    }
}
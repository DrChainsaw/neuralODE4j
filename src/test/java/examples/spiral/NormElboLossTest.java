package examples.spiral;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;

import java.util.Random;

import static org.junit.Assert.*;

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
    public void computeGradientAndScore() {
        final double sigma = 0.3;
        final NormElboLoss loss = new NormElboLoss(sigma, new NormElboLoss.ExtractQzZero() {
            @Override
            public Triple<INDArray, INDArray, INDArray> extractPredMeanLogvar(INDArray result) {
                return new Triple<>(
                        result.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 5)),
                        result.get(NDArrayIndex.all(), NDArrayIndex.point(5)),
                        result.get(NDArrayIndex.all(), NDArrayIndex.point(6)));
            }

            @Override
            public INDArray combinePredMeanLogvarEpsilon(INDArray predEps, INDArray meanEps, INDArray logvarEps) {
                return Nd4j.hstack(predEps, meanEps, logvarEps);
            }
        });

        final INDArray labels = Nd4j.create(new double[][]{
                {-1, 2, -3, 4, -5},
        });

        final INDArray resultConc = Nd4j.concat(1, labels.mul(3.45), Nd4j.scalar(1.23), Nd4j.scalar(2.34));

        final Pair<Double, INDArray> scoreAndGrad = loss.computeGradientAndScore(labels, resultConc, new ActivationIdentity(), null, false);
        assertEquals("Incorrect loss!",1836.949, scoreAndGrad.getFirst(), 1e-3); // Numerical precision with float on GPU vs CPU

        final double[] expectedGrad = {-27.22222137451172, 54.44444274902344, -81.66666412353516, 108.88888549804688, -136.11109924316406, 1.2300000190734863, 4.69061803817749};
        assertArrayEquals("Incorrect loss gradient!", expectedGrad, scoreAndGrad.getSecond().toDoubleVector(), 1e-5);
    }

    /**
     * Test that score array has correct dimension when a spiral is used as input
     */
    @Test
    public void computeScoreArraySpiral() {
        final double sigma = 0.3;
        final long nrofSamples = 50;
        final long nrofLatentDims = 4;
        final int batchSize = 10;

        final NormElboLoss loss = new NormElboLoss(sigma, new NormElboLoss.ExtractQzZero() {
            @Override
            public Triple<INDArray, INDArray, INDArray> extractPredMeanLogvar(INDArray result) {
                return new Triple<>(
                        result.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2*nrofSamples)),
                        result.get(NDArrayIndex.all(), NDArrayIndex.interval(2*nrofSamples, 2*nrofSamples + nrofLatentDims)),
                        result.get(NDArrayIndex.all(), NDArrayIndex.interval(2*nrofSamples+nrofLatentDims, 2*nrofSamples + 2*nrofLatentDims))
                );
            }

            @Override
            public INDArray combinePredMeanLogvarEpsilon(INDArray predEps, INDArray meanEps, INDArray logvarEps) {
                fail("Not expected!");
                return null;
            }
        });

        final SpiralIterator.Generator gen = new SpiralIterator.Generator(
                new SpiralFactory(0, 0.3, 0, 2*Math.PI, 200),
                0.3, nrofSamples, new Random(666));
        final INDArray labels = gen.generate(batchSize).getFeatures(0);

        final INDArray resultConc = Nd4j.hstack(
                labels.reshape(batchSize, labels.length() / batchSize),
                Nd4j.arange(batchSize*nrofLatentDims).reshape(batchSize, nrofLatentDims),
                Nd4j.arange(batchSize*nrofLatentDims).reshape(batchSize, nrofLatentDims));

        final INDArray score = loss.computeScoreArray(
                labels.reshape(batchSize, labels.length() / batchSize),
                resultConc, new ActivationIdentity(),
                null);

        assertEquals("Incorrect size!", batchSize, score.length());
    }
}
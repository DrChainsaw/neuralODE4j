package examples.spiral;

import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link SpiralIterator}
 *
 * @author Christian Skarby
 */
public class SpiralIteratorTest {

    /**
     * Test that a spiral can be generated into a {@link MultiDataSet} and that the dimensions are as expected
     */
    @Test
    public void next() {
        final long nrofSamplesOrig = 1000;
        final long nrofSamplesTrain = 100;
        final int batchSize = 200;
        final SpiralFactory spiralFactory = new SpiralFactory(0, 0.3, 0, 6*Math.PI, nrofSamplesOrig);
        final MultiDataSetIterator iterator = new SpiralIterator(
                new SpiralIterator.Generator(spiralFactory, 0.3, nrofSamplesTrain, new Random(666)),
                batchSize);

        final MultiDataSet mds = iterator.next();

        final long[] expectedShapeSpiral = {batchSize, 2, nrofSamplesTrain};
        assertArrayEquals("Incorrect shape of spiral!", expectedShapeSpiral, mds.getFeatures(0).shape());
        assertArrayEquals("Incorrect shape of label!", expectedShapeSpiral, mds.getLabels(0).shape());

        final long[] expectedShapeTime = {1, nrofSamplesTrain};
        assertArrayEquals("Incorrect shape of time!", expectedShapeTime, mds.getFeatures(1).shape());
    }
}
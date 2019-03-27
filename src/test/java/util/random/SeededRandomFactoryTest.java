package util.random;

import org.junit.Test;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

public class SeededRandomFactoryTest {

    /**
     * Test that the random seed can be reset to generate the same sequence again
     */
    @Test
    public void getRandom() {
        final long baseSeed = 666;
        final long[] shape = {2,3,4};
        SeededRandomFactory.setNd4jSeed(baseSeed);
        final Random first = Nd4j.getRandom();
        SeededRandomFactory.setNd4jSeed(baseSeed);
        final Random second = Nd4j.getRandom();

        assertEquals("Not same random!", first.nextGaussian(shape), second.nextGaussian(shape));
    }

    /**
     * Test that the new random instances
     */
    @Test
    public void getNewRandomInstance() {
        final long baseSeed = 666;
        final long[] shape = {2,3,4};
        SeededRandomFactory.setNd4jSeed(baseSeed);
        final Random first = Nd4j.getRandomFactory().getNewRandomInstance();
        SeededRandomFactory.setNd4jSeed(baseSeed);
        final Random second = Nd4j.getRandomFactory().getNewRandomInstance();

        assertEquals("Not same random!", first.nextGaussian(shape), second.nextGaussian(shape));
    }
}
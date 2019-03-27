package examples.spiral;

import org.junit.Test;

import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link SpiralFactory}
 *
 * @author Christian Skarby
 */
public class SpiralFactoryTest {

    /**
     * Test that spirals can be sampled from factory
     */
    @Test
    public void sample() {
        final SpiralFactory factory = new SpiralFactory(0, 0.3, 0, 10, 1000);
        final List<SpiralFactory.Spiral> sample = factory.sample(4, 100, () -> 0.3, () -> true);
        assertEquals("Incorrect number of samples!", 4, sample.size());
        for(SpiralFactory.Spiral spiral: sample) {
            assertArrayEquals("Incorrect trajectory!", new long[] {2, 100}, spiral.trajectory().shape());
            assertArrayEquals("Incorrect theta!", new long[] {1, 100}, spiral.theta().shape());
        }
    }
}
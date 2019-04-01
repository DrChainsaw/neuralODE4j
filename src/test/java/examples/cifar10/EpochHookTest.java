package examples.cifar10;

import org.deeplearning4j.optimize.api.TrainingListener;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link EpochHook}
 *
 * @author Christian Skarby
 */
public class EpochHookTest {

    private static class CountRunnable implements Runnable {

        private int cnt = 0;

        @Override
        public void run() {
            cnt++;
        }
    }

    /**
     * Test that callback happens when it is expected for a period of one
     */
    @Test
    public void iterationDonePeriodOne() {

        final CountRunnable probe = new CountRunnable();
        final TrainingListener listener = new EpochHook(1, probe);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 1, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 2, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 3, probe.cnt);

    }

    /**
     * Test that callback happens when it is expected for a period of three
     */
    @Test
    public void iterationDonePeriodThree() {

        final CountRunnable probe = new CountRunnable();
        final TrainingListener listener = new EpochHook(3, probe);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 0, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 0, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 1, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 1, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 1, probe.cnt);

        listener.onEpochEnd(null);
        assertEquals("Incorrect number of callbacks!", 2, probe.cnt);

    }
}
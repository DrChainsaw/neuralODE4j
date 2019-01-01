package util.time;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Test cases for {@link NanoTimer}
 *
 * @author Christian Skarby
 */
public class NanoTimerTest {

    /**
     * Test that timer can measure time properly
     */
    @Test
    public void startStop() throws InterruptedException {
        final long expected = 100 * 1000*1000;
        final long delta = 30 * 1000*1000;

        final WallClockTimer timer = new NanoTimer(duration -> {
            assertTrue("Timer measured too long!", duration < expected + delta);
            assertTrue("Timer measured too short!", duration > expected - delta);
        });

        timer.start();
        Thread.sleep(100);
        timer.stop();
    }

    /**
     * Test that timer can measure time properly
     */
    @Test
    public void startPauseStop() throws InterruptedException {
        final long expected = 100 * 1000*1000;
        final long delta = 50 * 1000*1000;

        final WallClockTimer timer = new NanoTimer(duration -> {
            assertTrue("Timer measured too long: " + duration + "!", duration < expected + delta);
            assertTrue("Timer measured too short: " + duration + "!", duration > expected - delta);
        });

        timer.start();
        Thread.sleep(30);
        timer.pause();
        Thread.sleep(50); // Happens during pause
        timer.start();
        Thread.sleep(30);
        timer.pause();
        Thread.sleep(50); // Happens during pause
        timer.start();
        Thread.sleep(40);
        timer.stop();
    }

}
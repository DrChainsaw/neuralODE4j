package util.time;

import org.junit.Test;

import java.util.DoubleSummaryStatistics;
import java.util.function.DoubleConsumer;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link StatisticsTimer}
 *
 * @author Christian Skarby
 */
public class StatisticsTimerTest {

    /**
     * Test that start and stop works
     */
    @Test
    public void startStop() {
        final DoubleSummaryStatistics statistics = new DoubleSummaryStatistics();
        final WallClockTimer timer = new StatisticsTimer(new MockTimer(statistics, 10, 20), statistics);

        timer.start();
        timer.stop();
        assertEquals("Incorrect statistics!", 10d, statistics.getAverage(), 1e-10);

        timer.start();
        timer.stop();
        assertEquals("Incorrect statistics!", 15d, statistics.getAverage(), 1e-10);
    }

    private static class MockTimer implements WallClockTimer {

        private final double[] times;
        private final DoubleConsumer consumer;
        private int cnt;

        private MockTimer( DoubleConsumer consumer, double... times) {
            this.times = times;
            this.consumer = consumer;
        }

        @Override
        public WallClockTimer start() {
            return this;
        }

        @Override
        public WallClockTimer pause() {
            return this;
        }

        @Override
        public WallClockTimer stop() {
            consumer.accept(times[cnt % times.length]);
            cnt++;
            return this;
        }
    }
}
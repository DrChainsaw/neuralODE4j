package util.time;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.DoubleConsumer;

/**
 * Measures time using System.nanoTime
 *
 * @author Christian Skarby
 */
public class NanoTimer implements WallClockTimer {

    private static final Logger log = LoggerFactory.getLogger(NanoTimer.class);

    private final DoubleConsumer timeConsumer;

    private long start;
    private long duration;

    public NanoTimer() {
        this(duration -> log.info("Duration: " + duration + " ns"));
    }

    public NanoTimer(DoubleConsumer timeConsumer) {
        this.timeConsumer = timeConsumer;
    }

    @Override
    public WallClockTimer start() {
        start = System.nanoTime();
        return this;
    }

    @Override
    public WallClockTimer pause() {
        duration += System.nanoTime() - start;
        return this;
    }

    @Override
    public WallClockTimer stop() {
        pause();
        timeConsumer.accept(duration);
        duration = 0;
        return this;
    }

    public static DoubleConsumer toMs(DoubleConsumer msConsumer) {
        return duration -> msConsumer.accept(duration * 1e-6);
    }
}

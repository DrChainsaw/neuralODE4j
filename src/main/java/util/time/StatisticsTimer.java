package util.time;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.DoubleSummaryStatistics;

/**
 * Measures time statistics
 *
 * @author Christian Skarby
 */
public class StatisticsTimer implements WallClockTimer {

    private static final Logger log = LoggerFactory.getLogger(StatisticsTimer.class);


    private final WallClockTimer sampleTimer;
    private final DoubleSummaryStatistics statistics;

    public StatisticsTimer() {
        this.statistics = new DoubleSummaryStatistics();
        this.sampleTimer = new NanoTimer(NanoTimer.toMs(this.statistics));
    }

    public StatisticsTimer(DoubleSummaryStatistics statistics) {
        this(new NanoTimer(NanoTimer.toMs(statistics)), statistics);
    }

    public StatisticsTimer(WallClockTimer sampleTimer, DoubleSummaryStatistics statistics) {
        this.sampleTimer = sampleTimer;
        this.statistics = statistics;
    }

    @Override
    public WallClockTimer start() {
        sampleTimer.start();
        return this;
    }

    @Override
    public WallClockTimer pause() {
        sampleTimer.pause();
        return this;
    }

    @Override
    public WallClockTimer stop() {
        sampleTimer.stop();
        return this;
    }

    public void logMean(String prefix) {
        log.info(prefix + " mean duration: " + statistics.getAverage() + " ms");
    }

    public void logMax(String prefix) {
        log.info(prefix + " mean duration: " + statistics.getMax() + " ms");
    }

    public void logMin(String prefix) {
        log.info(prefix + " mean duration: " + statistics.getMin()+ " ms");
    }
}

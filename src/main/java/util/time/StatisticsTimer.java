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
    public StatisticsTimer start() {
        sampleTimer.start();
        return this;
    }

    @Override
    public StatisticsTimer pause() {
        sampleTimer.pause();
        return this;
    }

    @Override
    public StatisticsTimer stop() {
        sampleTimer.stop();
        return this;
    }

    /**
     * Log the measured mean value.
     * @param prefix Prefix of log string
     */
    public void logMean(String prefix) {
        log.info(prefix + " mean duration: " + statistics.getAverage() + " ms");
    }

    /**
     * Log the measured max value.
     * @param prefix Prefix of log string
     */
    public void logMax(String prefix) {
        log.info(prefix + " max duration: " + statistics.getMax() + " ms");
    }

    /**
     * Log the measured min value.
     * @param prefix Prefix of log string
     */
    public void logMin(String prefix) {
        log.info(prefix + " min duration: " + statistics.getMin()+ " ms");
    }

    /**
     * Log the measured sum value.
     * @param prefix Prefix of log string
     */
    public void logSum(String prefix) {
        log.info(prefix + " sum duration: " + statistics.getSum() + " ms");
    }
}

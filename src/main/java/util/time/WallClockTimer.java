package util.time;

/**
 * Measurement of wall clock time for performance measurement
 *
 * @author Christian Skarby
 */
public interface WallClockTimer {

    /**
     * Start the timer
     * @return the WallClockTimer instance for fluent API
     */
    WallClockTimer start();

    /**
     * Pause the timer. It may be resumed though the resume method
     * @return the WallClockTimer instance for fluent API
     */
    WallClockTimer pause();

    /**
     * Stop the timer, resetting the timer
     * @return the WallClockTimer instance for fluent API
     */
    WallClockTimer stop();

}

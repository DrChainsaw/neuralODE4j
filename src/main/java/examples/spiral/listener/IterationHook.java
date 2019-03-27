package examples.spiral.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

/**
 * Call a {@link Runnable} every iterPeriod number of iterations.
 *
 * @author Christian Skarby
 */
public class IterationHook extends BaseTrainingListener {

    private final int iterPeriod;
    private final Runnable callback;

    public IterationHook(int iterPeriod, Runnable callback) {
        this.iterPeriod = iterPeriod;
        this.callback = callback;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(iteration > 0 && iteration % iterPeriod == 0) {
            callback.run();
        }
    }
}

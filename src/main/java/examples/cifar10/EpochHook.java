package examples.cifar10;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

/**
 * Call a {@link Runnable} every epochPeriod number of iterations.
 *
 * @author Christian Skarby
 */
public class EpochHook extends BaseTrainingListener {

    private final int epochPeriod;
    private final Runnable callback;

    private int epochCnt = 0;

    public EpochHook(int epochPeriod, Runnable callback) {
        this.epochPeriod = epochPeriod;
        this.callback = callback;
    }

    @Override
    public void onEpochEnd(Model model) {
        epochCnt++;
        if(epochCnt == epochPeriod) {
            callback.run();
            epochCnt = 0;
        }

    }
}
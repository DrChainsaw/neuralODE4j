package util.listen.step;

import ode.solve.api.StepListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Counts the number of steps taken
 *
 * @author Christian Skarby
 */
public class StepCounter implements StepListener {

    private static final Logger log = LoggerFactory.getLogger(StepCounter.class);

    private final int nrToAccum;
    private final ResultConsumer consumer;
    private int nrofSteps;
    private int nrofSolves;

    public interface ResultConsumer {
        void accept(int nrofSteps, int nrofSolves);
    }

    public StepCounter(int nrToAvergage) {
        this(nrToAvergage, "average number of solver steps: ");
    }

    public StepCounter(int nrToAccum, String msg) {
        this(nrToAccum, new ResultConsumer() {
            @Override
            public void accept(int nrofSteps, int nrofSolves) {
                log.info(msg + (double)nrofSteps / nrofSolves);
            }
        });
    }

    public StepCounter(int nrToAccum, ResultConsumer consumer) {
        this.nrToAccum = nrToAccum;
        this.consumer = consumer;
    }

    @Override
    public void begin(INDArray t, INDArray y0) {
        nrofSolves++;
    }

    @Override
    public void step(INDArray currTime, INDArray step, INDArray error, INDArray y) {
        nrofSteps++;
    }

    @Override
    public void done() {
        if(nrofSolves == nrToAccum) {
            consumer.accept(nrofSteps, nrofSolves);
            nrofSolves = 0;
            nrofSteps = 0;
        }
    }
}

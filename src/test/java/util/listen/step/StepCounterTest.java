package util.listen.step;

import ode.solve.api.StepListener;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link StepCounter}
 *
 * @author Christian Skarby
 */
public class StepCounterTest {

    /**
     * Test that counting and reporting happens after the given number of solves and that the reported number of steps
     * and solves is correct
     */
    @Test
    public void countAndReport() {
        final Probe probe = new Probe();
        StepListener listener = new StepCounter(3, probe);

        for(int i = 0; i < 7; i++) {
            listener.begin(Nd4j.linspace(0, 1, 2), Nd4j.zeros(1));
            listener.step(Nd4j.ones(1), Nd4j.ones(1), Nd4j.zeros(1), Nd4j.zeros(0));
            listener.step(Nd4j.ones(1), Nd4j.ones(1), Nd4j.zeros(1), Nd4j.zeros(0));
            listener.done();

            assertEquals("Incorrect number of calls!", (i+1) / 3, probe.nrofCalls);
        }
        assertEquals("Incorrect number of steps!", 3*2, probe.lastNrofSteps);
        assertEquals("Incorrect number of solves!", 3, probe.lastNrofSolves);
    }

    private static class Probe implements StepCounter.ResultConsumer {

        private int nrofCalls = 0;
        private int lastNrofSteps;
        private int lastNrofSolves;

        @Override
        public void accept(int nrofSteps, int nrofSolves) {
            nrofCalls++;
            lastNrofSteps = nrofSteps;
            lastNrofSolves = nrofSolves;
        }
    }

}
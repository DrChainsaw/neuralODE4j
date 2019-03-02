package ode.solve.impl.util;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import util.listen.step.ProbeStepListener;

/**
 * Test cases for {@link AggStepListener}
 *
 * @author Christian Skarby
 */
public class AggStepListenerTest {

    /**
     * Test that added listener are called and that a removed listener is no longer called
     */
    @Test
    public void addListenersAndRemoveOne() {
        final ProbeStepListener first = new ProbeStepListener();
        final ProbeStepListener second = new ProbeStepListener();
        final ProbeStepListener third = new ProbeStepListener();

        final AggStepListener aggStepListener = new AggStepListener();
        aggStepListener.addListeners(first);
        aggStepListener.addListeners(second,third);

        aggStepListener.begin(Nd4j.linspace(0,1 ,2), Nd4j.zeros(1));
        aggStepListener.step(new StateContainer(0, new double[] {0}, new double[] {0}), Nd4j.zeros(1), Nd4j.zeros(1));
        aggStepListener.done();
        first.assertNrofCalls(1,1,1);
        second.assertNrofCalls(1,1,1);
        third.assertNrofCalls(1,1,1);

        aggStepListener.clearListeners(second);

        aggStepListener.begin(Nd4j.linspace(0,1 ,2), Nd4j.zeros(1));
        aggStepListener.step(new StateContainer(0, new double[] {0}, new double[] {0}), Nd4j.zeros(1), Nd4j.zeros(1));
        aggStepListener.done();
        first.assertNrofCalls(2,2,2);
        second.assertNrofCalls(1,1,1);
        third.assertNrofCalls(2,2,2);

        aggStepListener.clearListeners(third, first);

        aggStepListener.begin(Nd4j.linspace(0,1 ,2), Nd4j.zeros(1));
        aggStepListener.step(new StateContainer(0, new double[] {0}, new double[] {0}), Nd4j.zeros(1), Nd4j.zeros(1));
        aggStepListener.done();
        first.assertNrofCalls(2,2,2);
        second.assertNrofCalls(1,1,1);
        third.assertNrofCalls(2,2,2);

        aggStepListener.addListeners(first, second, third);
        aggStepListener.clearListeners();

        aggStepListener.begin(Nd4j.linspace(0,1 ,2), Nd4j.zeros(1));
        aggStepListener.step(new StateContainer(0, new double[] {0}, new double[] {0}), Nd4j.zeros(1), Nd4j.zeros(1));
        aggStepListener.done();
        first.assertNrofCalls(2,2,2);
        second.assertNrofCalls(1,1,1);
        third.assertNrofCalls(2,2,2);
    }
}
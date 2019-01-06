package ode.solve.conf;

import ode.solve.api.FirstOrderSolverConf;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import util.listen.step.ProbeStepListener;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public abstract class AbstractConfTest {

    protected abstract FirstOrderSolverConf createConf();

    /**
     * Test serialization and deserialization
     */
    @Test
    public void serializeDeserialize() throws IOException {
        final FirstOrderSolverConf conf = createConf();
        final String json = NeuralNetConfiguration.mapper().writeValueAsString(conf);
        final FirstOrderSolverConf newConf = NeuralNetConfiguration.mapper().readValue(json, FirstOrderSolverConf.class);
        assertEquals("Did not deserialize into the same thing!", conf, newConf);
    }

    /**
     * Test that listeners are called correctly
     */
    @Test
    public void listenerCallback() {
        final FirstOrderSolverConf conf = createConf();
        final ProbeStepListener probeStepListener = new ProbeStepListener();
        final ProbeStepListener probeStepListenerRemove = new ProbeStepListener();
        final ProbeEquation probeEquation = new ProbeEquation();
        conf.addListeners(probeStepListener, probeStepListenerRemove);
        conf.clearListeners(probeStepListenerRemove);
        conf.instantiate().integrate(probeEquation, Nd4j.linspace(0,1,2), Nd4j.zeros(1), Nd4j.create(1));
        probeEquation.assertWasCalled();
        probeStepListener.assertWasCalled();
        probeStepListenerRemove.assertNrofCalls(0,0,0);
    }
}

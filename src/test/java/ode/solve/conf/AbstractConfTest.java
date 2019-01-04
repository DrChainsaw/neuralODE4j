package ode.solve.conf;

import ode.solve.api.FirstOrderSolverConf;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;

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
}

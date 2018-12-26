package examples.mnist;

import ode.solve.impl.DummyIteration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link OdeNetModel}
 *
 * @author Christian Skarby
 */
public class OdeNetModelTest {

    /**
     * Test that the model can be created and that it is possible to make train for two examples
     */
    @Test
    public void fit() {
        final ComputationGraph model = new OdeNetModel().create(new DummyIteration(() -> 3));
        model.fit(new DataSet(Nd4j.randn(1, 28*28), Nd4j.randn(1,10)));
        model.fit(new DataSet(Nd4j.randn(1, 28*28), Nd4j.randn(1,10)));
    }
}
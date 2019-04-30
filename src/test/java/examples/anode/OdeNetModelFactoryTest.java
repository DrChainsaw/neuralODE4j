package examples.anode;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class OdeNetModelFactoryTest {

    /**
     * Test that the model can be created and that it is possible to train for two examples
     */
    @Test
    public void fit1DInput() {
        final OdeNetModelFactory factory = new OdeNetModelFactory();

        final ComputationGraph model = factory.create(1).graph();
        model.fit(new DataSet(Nd4j.ones(1,1), Nd4j.ones(1,1).negi()));
        model.fit(new DataSet(Nd4j.ones(1,1).negi(), Nd4j.ones(1,1)));
    }
}
package examples.spiral;

import com.beust.jcommander.JCommander;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link OdeNetModel}
 *
 * @author Christian Skarby
 */
public class OdeNetModelTest {

    /**
     * Test that the model can be created and that it is possible to make train for two batches
     */
    @Test
    public void fit() {
        final OdeNetModel factory = new OdeNetModel();
        JCommander.newBuilder()
                .addObject(factory)
                .build()
                .parse();

        final long nrofTimeSteps = 10;
        final long batchSize = 3;
        final ComputationGraph model = factory.create(10, 0.3);
        model.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)},
                new INDArray[]{Nd4j.randn(batchSize,2 * nrofTimeSteps)}));
        model.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)},
                new INDArray[]{Nd4j.randn(batchSize,2 * nrofTimeSteps)}));
    }

}
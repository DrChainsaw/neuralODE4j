package examples.mnist;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link ResNetReferenceModel}
 *
 * @author Christian Skarby
 */
public class ResNetReferenceModelTest {

    /**
     * Test that the model can be created and that it is possible to make one forward pass.
     */
    @Test
    public void create() {
        final ComputationGraph model = new ResNetReferenceModel().create();
        model.output(Nd4j.randn(new long[] {1, 1, 28,28}));
    }
}
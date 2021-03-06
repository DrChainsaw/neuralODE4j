package examples.mnist;

import com.beust.jcommander.JCommander;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link ResNetReferenceModel}
 *
 * @author Christian Skarby
 */
public class ResNetReferenceModelTest {

    /**
     * Test that the model can be created and that it is possible to make train for two examples using the "res" stem
     */
    @Test
    public void fitResStem() {
        final ModelFactory factory = new ResNetReferenceModel();
        JCommander.newBuilder()
                .addObject(factory)
                .build()
                .parse("-stem", "res");

        final ComputationGraph model = factory.create();
        model.fit(new DataSet(Nd4j.randn(new long[]{1, 1, 28, 28}), Nd4j.randn(1,10)));
        model.fit(new DataSet(Nd4j.randn(new long[]{1, 1, 28, 28}), Nd4j.randn(1,10)));
    }

    /**
     * Test that the model can be created and that it is possible to make train for two examples using the "conv" stem
     */
    @Test
    public void fitConvStem() {
        final ModelFactory factory = new ResNetReferenceModel();
        JCommander.newBuilder()
                .addObject(factory)
                .build()
                .parse("-stem", "conv");

        final ComputationGraph model = factory.create();
        model.fit(new DataSet(Nd4j.randn(new long[]{1, 1, 28, 28}), Nd4j.randn(1,10)));
        model.fit(new DataSet(Nd4j.randn(new long[]{1, 1, 28, 28}), Nd4j.randn(1,10)));
    }
}
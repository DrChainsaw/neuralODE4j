package examples.spiral;

import com.beust.jcommander.JCommander;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.fail;

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
        final ComputationGraph model = factory.create(nrofTimeSteps, 0.3, 3);
        model.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)},
                new INDArray[]{Nd4j.randn(batchSize, 2 * nrofTimeSteps)}));
        model.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)},
                new INDArray[]{Nd4j.randn(batchSize, 2 * nrofTimeSteps)}));
    }

    /**
     * Test that the model can be serialized and deserialized into the same thing
     */
    @Test
    public void testSerializeDeserialize() throws IOException {
        final OdeNetModel factory = new OdeNetModel();
        final int nrofTimeSteps = 5;
        JCommander.newBuilder()
                .addObject(factory)
                .build()
                .parse();
        final ComputationGraph graph = factory.create(nrofTimeSteps, 0.1, 4);

        final Path baseDir = Paths.get("src", "test", "resources", "OdeNetModelTest");
        final String fileName = Paths.get(baseDir.toString(), "testSerializeDeserialize.zip").toString();

        try {

            baseDir.toFile().mkdirs();
            graph.save(new File(fileName), true);
            final ComputationGraph newGraph = ModelSerializer.restoreComputationGraph(new File(fileName), true);

            assertEquals("Config was not restored properly!", graph.getConfiguration(), newGraph.getConfiguration());

            final long batchSize = 3;
            final INDArray[] input = {Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)};

            assertEquals("Output not the same!", graph.outputSingle(input), newGraph.outputSingle(input));

        } catch (IOException e) {
            e.printStackTrace();
            fail("Failed to serialize or deserialize graph!");
        } finally {
            new File(fileName).delete();
            Files.delete(baseDir);
        }
    }
}

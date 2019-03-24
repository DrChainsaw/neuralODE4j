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
import static org.junit.Assert.assertArrayEquals;
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

        final long nrofTimeSteps = 10;
        final long batchSize = 3;
        final long nrofLatentDims = 5;
        final ComputationGraph model = factory.createNew(nrofTimeSteps, 0.3, nrofLatentDims).trainingModel();
        model.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)},
                new INDArray[]{Nd4j.randn(new long[] {batchSize, 2, nrofTimeSteps}), Nd4j.zeros(batchSize, nrofLatentDims)}));
        model.fit(new MultiDataSet(
                new INDArray[]{Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)},
                new INDArray[]{Nd4j.randn(new long[] {batchSize, 2, nrofTimeSteps}), Nd4j.zeros(batchSize, nrofLatentDims)}));
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
        final ComputationGraph graph = factory.createNew(nrofTimeSteps, 0.1, 4).trainingModel();

        final Path baseDir = Paths.get("src", "test", "resources", "OdeNetModelTest");
        final String fileName = Paths.get(baseDir.toString(), "testSerializeDeserialize.zip").toString();

        try {

            baseDir.toFile().mkdirs();
            graph.save(new File(fileName), true);
            final ComputationGraph newGraph = ModelSerializer.restoreComputationGraph(new File(fileName), true);

            assertEquals("Config was not restored properly!", graph.getConfiguration(), newGraph.getConfiguration());

            final long batchSize = 3;
            final INDArray[] input = {Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps}), Nd4j.linspace(0, 3, nrofTimeSteps)};

            assertEquals("Output not the same!", graph.output(input)[0], newGraph.output(input)[0]);

        } catch (IOException e) {
            e.printStackTrace();
            fail("Failed to serialize or deserialize graph!");
        } finally {
            new File(fileName).delete();
            Files.delete(baseDir);
        }
    }

    /**
     * Smoke test to assert that an {@link OdeNetModel} can be represented as a {@link TimeVae} without there being
     * any exceptions.
     */
    @Test
    public void asTimeVae()  {
        final long batchSize = 5;
        final long nrofTimeSteps = 17;
        final long nrofLatentDims = 6;

        final TimeVae timeVae = new OdeNetModel().createNew(nrofTimeSteps, 0.3, nrofLatentDims);

        final INDArray inputTraj = Nd4j.randn(new long[]{batchSize, 2, nrofTimeSteps});
        final INDArray time =  Nd4j.linspace(0, 3, nrofTimeSteps);

        final INDArray z0 = timeVae.encode(inputTraj);

        assertArrayEquals("Incorrect shape of z0!", new long[] {batchSize, nrofLatentDims}, z0.shape());

        final INDArray zt = timeVae.timeDependency(z0, time);

        assertArrayEquals("Incorrect shape of zt!", new long[] {batchSize, nrofLatentDims, nrofTimeSteps}, zt.shape());

        final INDArray decoded = timeVae.decode(zt);

        assertArrayEquals("Incorrect shape of decoded output!", new long[] {batchSize, 2, nrofTimeSteps}, decoded.shape());

    }
}

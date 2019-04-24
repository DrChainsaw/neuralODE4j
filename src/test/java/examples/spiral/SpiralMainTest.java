package examples.spiral;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link Main}
 *
 * @author Christian Skarby
 */
public class SpiralMainTest {

    /**
     * Test that a model can be created from scratch, trained for two epochs and then resumed from the last checkpoint
     */
    @Test
    public void main() throws IOException {

        final Path baseDir = Paths.get("src", "test", "resources", "testSpiralMain");
        try {
            final int nrofTrainIters = 2;
            final Main main = runNewMain(baseDir, nrofTrainIters);

            final Path checkpoint = Paths.get(main.saveDir(), Main.CHECKPOINT_NAME);
            assertTrue("Checkpoint not saved!", checkpoint.toFile().exists());

            final int newNrofTrainIters = 4;
            runNewMain(baseDir, newNrofTrainIters);

            assertTrue("Checkpoint not saved!", checkpoint.toFile().exists());

            final ComputationGraph model = ModelSerializer.restoreComputationGraph(checkpoint.toFile());

            assertEquals("Incorrect epoch!", newNrofTrainIters, model.getIterationCount());


        } catch (IOException e) {
            e.printStackTrace();
            fail("Failed to serialize or deserialize graph!");
        } finally {
            FileUtils.deleteDirectory(new File(baseDir.toString()));
        }

    }

    private Main runNewMain(Path baseDir, int nrofTrainIters) throws IOException {
        final Main main = new Main();
        final ModelFactory factory = Main.parseArgs(main,
                "-nrofTrainIters", String.valueOf(nrofTrainIters),
                "-saveDir", baseDir.toString(),
                "-trainBatchSize", "2",
                "-nrofTimeStepsForTraining", "10",
                "-saveEveryNIterations", "1",
                "-noPlot",
                "odenet");

        Main.createModel(main, factory);
        main.addListeners();
        main.run();
        return main;
    }
    
}
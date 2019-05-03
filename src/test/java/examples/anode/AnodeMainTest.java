package examples.anode;

import org.apache.commons.io.FileUtils;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Main}
 *
 * @author Christian Skarby
 */
public class AnodeMainTest {


    /**
     * Test that a model can be created from scratch and trained for two epochs for 1D case.
     */
    @Test
    public void main1D() throws IOException {
        final Path baseDir = Paths.get("src", "test", "resources", "testAnodeMain1D");
        try {
            final int nrofEpochs = 2;
            final Main main = runNewMain(nrofEpochs, baseDir, "-separable");
            assertEquals("Incorrect number of epochs!", nrofEpochs, main.model.graph().getEpochCount());
        } finally {
            FileUtils.deleteDirectory(new File(baseDir.toString()));
        }
    }

    /**
     * Test that a model can be created from scratch and trained for two epochs for 2D case.
     */
    @Test
    public void main2D() throws IOException {
        final Path baseDir = Paths.get("src", "test", "resources", "testAnodeMain2D");
        try {
            final int nrofEpochs = 2;
            final Main main = runNewMain(nrofEpochs, baseDir, "-2D");
            assertEquals("Incorrect number of epochs!", nrofEpochs, main.model.graph().getEpochCount());
        } finally {
            FileUtils.deleteDirectory(new File(baseDir.toString()));
        }
    }

    private Main runNewMain(int nrofEpochs, Path baseDir, String preArg) {
        final Main main = Main.parseArgs(preArg,
                "-plotsOff",
                "-nrofEpochs", String.valueOf(nrofEpochs),
                "-saveDir", baseDir.toString(),
                "-nrofExamples", "64", "-trainBatchSize", "64", "odenet");

        main.addListeners();
        main.run();

        return main;
    }
}
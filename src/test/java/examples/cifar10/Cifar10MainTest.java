package examples.cifar10;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.SingletonDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.factory.Nd4j;

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
public class Cifar10MainTest {

    /**
     * Test that a model can be created from scratch, trained for two epochs and then resumed from the last checkpoint
     */
    @Test
    public void main() throws IOException {

        final Path baseDir = Paths.get("src", "test", "resources", "testCifar10Main");
        try {
            final int nrofEpochs = 2;
            final Main main = runNewMain(baseDir, nrofEpochs);

            final Path checkpoint = Paths.get(main.saveDir(), Main.CHECKPOINT_NAME);
            assertTrue("Checkpoint not saved!", checkpoint.toFile().exists());

            final int newNrofEpochs = 4;
            runNewMain(baseDir, newNrofEpochs);

            assertTrue("Checkpoint not saved!", checkpoint.toFile().exists());

            final ComputationGraph model = ModelSerializer.restoreComputationGraph(checkpoint.toFile());

            assertEquals("Incorrect epoch!", newNrofEpochs, model.getEpochCount());


        } catch (IOException e) {
            e.printStackTrace();
            fail("Failed to serialize or deserialize graph!");
        } finally {
            FileUtils.deleteDirectory(new File(baseDir.toString()));
        }

    }

    private Main runNewMain(Path baseDir, int nrofEpochs) throws IOException {
        final Main main = Main.parseArgs("-nrofEpochs", String.valueOf(nrofEpochs), "-saveDir", baseDir.toString(), "odenet", "-useABlock", "-nrofPreReductions", "3");
        main.trainFactory = new DummyDataSetIterFactory();
        main.evalFactory = new DummyDataSetIterFactory();

        main.addListeners();
        main.run();
        return main;
    }

    private static class DummyDataSetIterFactory implements DataSetIteratorFactory {


        @Override
        public DataSetIterator create() {
            return new SingletonDataSetIterator(new DataSet(Nd4j.randn(new long[]{1, 3, 32, 32}), Nd4j.ones(1, 10)));
        }
    }
}
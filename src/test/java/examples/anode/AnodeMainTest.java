package examples.anode;

import org.junit.Test;
import util.plot.NoPlot;

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
    public void main1D() {

        final int nrofEpochs = 2;
        final Main main = runNewMain(nrofEpochs, "-separable");
        assertEquals("Incorrect number of epochs!", nrofEpochs, main.model.graph().getEpochCount());
    }

    /**
     * Test that a model can be created from scratch and trained for two epochs for 2D case.
     */
    @Test
    public void main2D() {

        final int nrofEpochs = 2;
        final Main main = runNewMain(nrofEpochs, "-2D");
        assertEquals("Incorrect number of epochs!", nrofEpochs, main.model.graph().getEpochCount());
    }

    private Main runNewMain(int nrofEpochs, String preArg) {
        final Main main = Main.parseArgs(preArg, "-nrofEpochs", String.valueOf(nrofEpochs), "-nrofExamples", "64", "-trainBatchSize", "64", "odenet");

        main.plotFactory = new NoPlot.Factory();
        main.addListeners();
        main.run();

        return main;
    }
}
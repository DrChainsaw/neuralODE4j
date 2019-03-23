package examples.spiral;

import lombok.val;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.*;

/**
 * Test cases for {@link FfToRnnNoPermute} and {@link RnnToFfNoPermute}
 *
 * @author Christian Skarby
 */
public class FfToRnnNoPermuteTest {

    private final static LayerWorkspaceMgr wsmgr = LayerWorkspaceMgr.noWorkspacesImmutable();

    @Test
    public void preProcess() {
        final int batchSize = 3;
        final long nrofFeatures = 5;
        final long nrofTimeSteps = 7;
        final INDArray input =
                getOrdered3dFrom1d(Nd4j.arange(batchSize * nrofFeatures * nrofTimeSteps),
                        batchSize, nrofFeatures, nrofTimeSteps);

        final INDArray output =
                new FfToRnnNoPermute().preProcess(
                        new RnnToFfNoPermute().preProcess(
                                input.dup(),
                                batchSize, wsmgr),
                        batchSize, wsmgr);

        assertEquals("Preprocessor op not inverted", input, output);
    }

    @Test
    public void backprop() {
        final int batchSize = 3;
        final long nrofFeatures = 5;
        final long nrofTimeSteps = 7;
        final INDArray input =
                getOrdered2dFrom1d(Nd4j.arange(batchSize * nrofFeatures * nrofTimeSteps),
                        batchSize, nrofFeatures, nrofTimeSteps);

        final INDArray output =
                new FfToRnnNoPermute().backprop(
                        new RnnToFfNoPermute().backprop(
                                input.dup(),
                                batchSize, wsmgr),
                        batchSize, wsmgr);

        assertEquals("Preprocessor op not inverted", input, output);
    }

    @Test
    public void e2e() {
        final int batchSize = 3;
        final long nrofFeatures = 5;
        final long nrofTimeSteps = 7;
        final INDArray input =
                getOrdered3dFrom1d(Nd4j.arange(batchSize * nrofFeatures * nrofTimeSteps),
                        batchSize, nrofFeatures, nrofTimeSteps);

        final InputPreProcessor rnnToFf = new RnnToFfNoPermute();
        final InputPreProcessor ffToRnn = new FfToRnnNoPermute();

        final INDArray output =
                rnnToFf.backprop(
                        ffToRnn.backprop(
                                ffToRnn.preProcess(
                                        rnnToFf.preProcess(
                                                input.dup(),
                                                batchSize, wsmgr),
                                        batchSize, wsmgr),
                                batchSize, wsmgr),
                        batchSize, wsmgr);

        assertEquals("Preprocessor op not inverted", input, output);
    }

    @Test
    public void e2eBatch1() {
        final int batchSize = 1;
        final long nrofFeatures = 3;
        final long nrofTimeSteps = 5;
        final INDArray input =
                getOrdered3dFrom1d(Nd4j.arange(batchSize * nrofFeatures * nrofTimeSteps),
                        batchSize, nrofFeatures, nrofTimeSteps);

        final InputPreProcessor rnnToFf = new RnnToFfNoPermute();
        final InputPreProcessor ffToRnn = new FfToRnnNoPermute();

        final INDArray output =
                rnnToFf.backprop(
                        ffToRnn.backprop(
                                ffToRnn.preProcess(
                                        rnnToFf.preProcess(
                                                input.dup(),
                                                batchSize, wsmgr),
                                        batchSize, wsmgr),
                                batchSize, wsmgr),
                        batchSize, wsmgr);
        assertEquals("Preprocessor op not inverted", input, output);
    }


    @Test
    public void RnnToFfOrderingPreProcess() {
        final int batchSize = 3;
        final long nrofFeatures = 5;
        final long nrofTimeSteps = 7;
        final INDArray input = Nd4j.arange(batchSize * nrofFeatures * nrofTimeSteps);

        assertEquals("Incorrect output!",
                getOrdered2dFrom1d(input.dup(), batchSize, nrofFeatures, nrofTimeSteps),
                new RnnToFfNoPermute().preProcess(
                        getOrdered3dFrom1d(input.dup(), batchSize, nrofFeatures, nrofTimeSteps), batchSize, wsmgr));

    }

    @Test
    public void RnnToFfOrderingBackprop() {
        final int batchSize = 3;
        final long nrofFeatures = 5;
        final long nrofTimeSteps = 7;
        final INDArray input = Nd4j.arange(batchSize * nrofFeatures * nrofTimeSteps);

        assertEquals("Incorrect output!",
                getOrdered3dFrom1d(input.dup(), batchSize, nrofFeatures, nrofTimeSteps),
                new RnnToFfNoPermute().backprop(
                        getOrdered2dFrom1d(input.dup(), batchSize, nrofFeatures, nrofTimeSteps), batchSize, wsmgr));

    }

    private static INDArray getOrdered3dFrom1d(INDArray array, long bs, long fs, long ts) {
        return array.reshape(bs, ts, fs).permutei(0, 2, 1);
    }

    private static INDArray getOrdered2dFrom1d(INDArray array, long bs, long fs, long ts) {
        return array.reshape(bs * ts, fs);
    }
}
package examples.spiral;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AddKLDLabel}
 */
public class AddKLDLabelTest {

    /**
     * Test that labels are added
     */
    @Test
    public void preProcess() {
        final long batchSize = 3;
        final long nrofLatentDims = 5;

        final MultiDataSet mds = new MultiDataSet(Nd4j.ones(batchSize, 13), Nd4j.ones(batchSize, 11));

        final MultiDataSetPreProcessor addKld = new AddKLDLabel(1.23, 2.34, nrofLatentDims);
        addKld.preProcess(mds);

        final INDArray kldLabel = mds.getLabels(1);

        assertArrayEquals("Incorrect shape!", new long[] {batchSize, 2*nrofLatentDims}, kldLabel.shape());

        assertEquals("Expected first half to be mean!", 1.23,
                kldLabel.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(0, nrofLatentDims)).meanNumber().doubleValue(), 1e-5);

        assertEquals("Expected second half to be log(var)!", Math.log(2.34),
                kldLabel.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(nrofLatentDims, 2*nrofLatentDims)).meanNumber().doubleValue(), 1e-5);

    }
}
package examples.spiral;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Adds a label for KLD loss
 *
 * @author Christian Skarby
 */
public class AddKLDLabel implements MultiDataSetPreProcessor {

    private final double mean;
    private final double var;
    private final long nrofLatentDims;

    public AddKLDLabel(double mean, double var, long nrofLatentDims) {
        this.mean = mean;
        this.var = var;
        this.nrofLatentDims = nrofLatentDims;
    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {
        final INDArray label0 = multiDataSet.getLabels(0);
        final long batchSize = label0.size(0);
        final INDArray kldLabel = Nd4j.ones(batchSize, 2*nrofLatentDims);
        kldLabel.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, nrofLatentDims)}, mean);
        kldLabel.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(nrofLatentDims, 2*nrofLatentDims)}, var);
        multiDataSet.setLabels(new INDArray[]{label0, kldLabel});
    }
}

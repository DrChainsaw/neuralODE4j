package examples.anode;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Collections;
import java.util.List;

/**
 * {@link DataSetIterator} for the function used in section 4.1 in https://arxiv.org/pdf/1904.01681.pdf
 * <br><br>
 * Provides input and output to the trivial function
 *
 * @author Christian Skarby
 */
public class AnodeToyIterator1D implements DataSetIterator {

    private final int batchSize;
    private final INDArray dataset;

    public AnodeToyIterator1D(int batchSize, INDArray dataset) {
        this.batchSize = batchSize;
        this.dataset = dataset;
    }

    @Override
    public DataSet next(int num) {
        final INDArray features = Nd4j.ones(num, 1);
        features.get(NDArrayIndex.interval(0, num/2), NDArrayIndex.all()).negi();
        return new DataSet(features, features.neg());
    }

    @Override
    public int inputColumns() {
        return 1;
    }

    @Override
    public int totalOutcomes() {
        return 1;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not supported!");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return Collections.singletonList("g_1d(x)");
    }

    @Override
    public boolean hasNext() {
        return true;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}

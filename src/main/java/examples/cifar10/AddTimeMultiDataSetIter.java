package examples.cifar10;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeMultiDataSetPreProcessor;

/**
 * Add time as input to a given {@link DataSetIterator}
 *
 * @author Christian Skarby
 */
class AddTimeMultiDataSetIter implements MultiDataSetIterator {

    private final DataSetIterator iterator;
    private final INDArray time;
    private MultiDataSetPreProcessor preProcessor = new CompositeMultiDataSetPreProcessor();

    AddTimeMultiDataSetIter(DataSetIterator iterator, INDArray time) {
        this.iterator = iterator;
        this.time = time;
    }

    @Override
    public MultiDataSet next(int num) {
        final DataSet ds = iterator.next(num);

        final MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{ds.getFeatures(), time.dup()},
                new INDArray[] {ds.getLabels()}
        );
        preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    public AddTimeMultiDataSetIter setPreProcessor(DataSetPreProcessor preProcessor) {
        iterator.setPreProcessor(preProcessor);
        return this;
    }

    @Override
    public boolean resetSupported() {
        return iterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return iterator.asyncSupported();
    }

    @Override
    public void reset() {
        iterator.reset();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public MultiDataSet next() {
        return next(iterator.batch());
    }
}

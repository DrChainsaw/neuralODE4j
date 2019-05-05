package examples.cifar10;

import com.beust.jcommander.Parameter;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

/**
 * {@link DataSetIteratorFactory} for CIFAR 10 test data set.
 *
 * @author Christian Skarby
 */
public class Cifar10TestDataProvider implements DataSetIteratorFactory {

    @Parameter(names = "-evalBatchSize", description = "Batch size to use for validation")
    private int evalBatchSize = 24;

    @Override
    public DataSetIterator create() {
        return new Cifar10DataSetIterator(
                evalBatchSize,
                DataSetType.TEST);
    }
}

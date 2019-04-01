package examples.cifar10;

import com.beust.jcommander.Parameter;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

/**
 * {@link DataSetIteratorFactory} for CIFAR 10 test data set.
 *
 * @author Christian Skarby
 */
public class Cifar10TestDataProvider implements DataSetIteratorFactory {

    @Parameter(names = "-evalBatchSize", description = "Batch size to use for validation")
    private int evalBatchSize = 32;

    @Parameter(names = "-nrofTestExamples", description = "Number of examples to use for validation")
    private int nrofTestExamples = CifarLoader.NUM_TEST_IMAGES;

    @Override
    public DataSetIterator create() {
        return new CifarDataSetIterator(
                evalBatchSize,
                nrofTestExamples,
                true);
    }
}

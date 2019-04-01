package examples.mnist;

import com.beust.jcommander.Parameter;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import util.preproc.Reshape;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * {@link DataSetIteratorFactory} for MNIST test data set.
 *
 * @author Christian Skarby
 */
public class MnistTestDataProvider implements DataSetIteratorFactory {

    @Parameter(names = "-evalBatchSize", description = "Batch size to use for validation")
    private int evalBatchSize = 1000;

    @Parameter(names = "-nrofTestExamples", description = "Number of examples to use for validation")
    private int nrofTestExamples = MnistDataFetcher.NUM_EXAMPLES_TEST;


    @Override
    public DataSetIterator create() {
        final DataSetIterator iter;
        try {
            iter = new MnistDataSetIterator(
                    evalBatchSize,
                    nrofTestExamples,
                    false, false, false, 1234) {
                @Override
                public DataSet next() {
                    // Original implementation does not apply preprocessor!
                    return next(evalBatchSize);
                }
            };

            iter.setPreProcessor(new Reshape(InputType.convolutional(28, 28, 1)));

            return iter;

        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}

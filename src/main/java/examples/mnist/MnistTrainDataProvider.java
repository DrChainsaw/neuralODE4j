package examples.mnist;

import com.beust.jcommander.Parameter;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import util.preproc.Reshape;
import util.preproc.ShiftDim;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Random;

/**
 * {@link DataSetIteratorFactory} for MNIST train data set.
 *
 * @author Christian Skarby
 */
public class MnistTrainDataProvider implements DataSetIteratorFactory {

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int trainBatchSize = 128;

    @Parameter(names = "-nrofTrainExamples", description = "Number of examples to use for training")
    private int nrofTrainExamples = MnistDataFetcher.NUM_EXAMPLES;

    @Parameter(names = "-data_aug", description = "Use data augmentation for training if set to true", arity = 1)
    private boolean useDataAugmentation = true;

    @Override
    public DataSetIterator create() {
        final DataSetIterator iter;
        try {
            iter = new MnistDataSetIterator(
                    trainBatchSize,
                    nrofTrainExamples,
                    false, true, true, 1234) {
                @Override
                public DataSet next() {
                    // Original implementation does not apply preprocessor!
                    return next(trainBatchSize);
                }
            };

            iter.setPreProcessor(new Reshape(InputType.convolutional(28, 28, 1)));

            if (useDataAugmentation) {
                iter.setPreProcessor(new CompositeDataSetPreProcessor(
                        iter.getPreProcessor(),
                        new ShiftDim(2, new Random(666), 4),
                        new ShiftDim(3, new Random(667), 4)
                ));
            }

            return iter;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

    }
}

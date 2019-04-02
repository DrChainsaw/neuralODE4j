package examples.cifar10;

import com.beust.jcommander.Parameter;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import util.preproc.ShiftDim;

import java.util.Random;

/**
 * {@link DataSetIteratorFactory} for CIFAR 10 train data set.
 *
 * @author Christian Skarby
 */
public class Cifar10TrainDataProvider implements DataSetIteratorFactory {

    @Parameter(names = "-trainBatchSize", description = "Batch size to use for training")
    private int trainBatchSize = 24;

    @Parameter(names = "-nrofTrainExamples", description = "Number of examples to use for training")
    private int nrofTrainExamples = CifarLoader.NUM_TRAIN_IMAGES;

    @Parameter(names = "-dataAug", description = "Use data augmentation for training if set to true", arity = 1)
    private boolean useDataAugmentation = true;

    @Override
    public DataSetIterator create() {
        final DataSetIterator iter =  new CifarDataSetIterator(
                trainBatchSize,
                nrofTrainExamples,
                true);

        if (useDataAugmentation) {
            iter.setPreProcessor(new CompositeDataSetPreProcessor(
                    new ShiftDim(2, new Random(666), 4),
                    new ShiftDim(3, new Random(667), 4)
            ));
        }

        return iter;
    }
}

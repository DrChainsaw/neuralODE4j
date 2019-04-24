package examples.cifar10;

import com.beust.jcommander.Parameter;
import org.datavec.image.loader.CifarLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

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
        final int numCh = CifarLoader.CHANNELS;
        final int height = CifarLoader.HEIGHT;
        final int width = CifarLoader.WIDTH;

        final ImageTransform dataAug = useDataAugmentation
                ? new PipelineImageTransform(
                667,
                Arrays.asList(
                        new Pair<>(new BoxImageTransform(width + 4, height + 4), 1d),
                        new Pair<>(new RandomCropTransform(666, height, width), 1d),
                        new Pair<>(new FlipImageTransform(1), 0.5d) // 50% chance of flip
                ))
                : new MultiImageTransform();

        return new CifarDataSetIterator(
                trainBatchSize,
                nrofTrainExamples,
                new int[]{height, width, numCh},
                CifarLoader.NUM_LABELS,
                dataAug,
                false,
                true,
                123,
                true);
    }
}

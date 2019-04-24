package examples.cifar10;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Interface for models
 *
 * @author Christian Skarby
 */
interface ModelFactory {

    /**
     * Create the model to use
     * @return a {@link ComputationGraph} for the model
     */
    ComputationGraph create();

    /**
     * Return the name of the model built to use e.g. for saving models
     * @return the name of the models
     */
    String name();

    /**
     * Wrap iterator in a suitable MultiDataSetIterator
     * @param iter Source data set iterator
     * @return MultiDataSetIterator to use
     */
    MultiDataSetIterator wrapIter(DataSetIterator iter);
}

package examples.mnist;

import org.deeplearning4j.nn.graph.ComputationGraph;

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
}

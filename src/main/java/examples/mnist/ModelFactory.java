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
}

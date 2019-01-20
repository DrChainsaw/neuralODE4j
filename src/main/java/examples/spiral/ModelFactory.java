package examples.spiral;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * Interface for models
 *
 * @author Christian Skarby
 */
public interface ModelFactory {

    /**
     * Create the model to use
     * @param nrofSamples The number of samples in each spiral
     * @return a {@link ComputationGraph} for the model
     */
    ComputationGraph create(long nrofSamples);

    /**
     * Return the name of the model built to use e.g. for saving models
     * @return the name of the models
     */
    String name();
}

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
     * @param noiseSigma Noise std for training spirals
     * @return a {@link ComputationGraph} for the model
     */
    ComputationGraph create(long nrofSamples, double noiseSigma);

    /**
     * Return the name of the model built to use e.g. for saving models
     * @return the name of the models
     */
    String name();
}

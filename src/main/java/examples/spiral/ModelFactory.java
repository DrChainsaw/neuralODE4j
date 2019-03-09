package examples.spiral;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;

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
     * @param nrofLatentDims How many dimensions for latent variable
     * @return a {@link ComputationGraph} for the model
     */
    ComputationGraph create(long nrofSamples, double noiseSigma, long nrofLatentDims);

    /**
     * Return the name of the model built to use e.g. for saving models
     * @return the name of the models
     */
    String name();

    /**
     * Return a {@link MultiDataSetPreProcessor} which needs to be applied to the input
     * @return a {@link MultiDataSetPreProcessor}
     */
    MultiDataSetPreProcessor getPreProcessor(long nrofLatentDims);
}

package examples.anode;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * Creates models for ANODE experiments
 *
 * @author Christian Skarby
 */
interface ModelFactory {

    /**
     * Create the model to use
     * @param nrofInputDims number of dimensions in input
     * @return a {@link ComputationGraph} for the model
     */
    ComputationGraph create(long nrofInputDims);
}

package examples.anode;

/**
 * Creates models for ANODE experiments
 *
 * @author Christian Skarby
 */
interface ModelFactory {

    /**
     * Create the model to use
     * @param nrofInputDims number of dimensions in input
     * @return a {@link Model}
     */
    Model create(long nrofInputDims);
}

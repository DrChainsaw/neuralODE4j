package ode.vertex.impl.gradview;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * Creates an {@link INDArray1DView} of all gradients in a set of vertices. Reason why this exists is that there are
 * instances of parameters which DL4J puts in the gradient view but which are not actually gradients. One example is the
 * running mean and variance of batchnorm. Such parameters does not play well with adjoint backpropagation for Neural
 * ODEs.
 *
 * @author Christian Skarby
 */
public interface GradientViewFactory {

    /**
     * Create a {@link INDArray1DView} of the gradients of the given graph.
     * @param graph Graph to extract gradients from
     * @return INDArray1DView of the gradients
     */
    INDArray1DView create(ComputationGraph graph);
}

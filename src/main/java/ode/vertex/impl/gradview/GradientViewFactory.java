package ode.vertex.impl.gradview;

import ode.vertex.impl.gradview.parname.ParamNameMapping;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * Creates an {@link INDArray1DView} of all gradients in a set of vertices. Reason why this exists is that there are
 * instances of parameters which DL4J puts in the gradient view but which are not actually gradients. One example is the
 * running mean and variance of batchnorm. Such parameters does not play well with adjoint backpropagation for Neural
 * ODEs.
 *
 * @author Christian Skarby
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface GradientViewFactory extends Serializable {

    /**
     * Create a {@link ParameterGradientView} of the gradients of the given graph.
     *
     * @param graph Graph to extract gradients from
     * @return Views of the gradients
     */
    ParameterGradientView create(ComputationGraph graph);

    /**
     * Return the mapping used to newPlot non-colliding names
     *
     * @return the mapping used to newPlot non-colliding names
     */
    ParamNameMapping paramNameMapping();

    /**
     * Clone the factory
     *
     * @return a clone
     */
    GradientViewFactory clone();
}

package ode.vertex.impl.gradview;

import org.deeplearning4j.nn.gradient.Gradient;

/**
 * Different views of the parameter gradients of a graph.
 *
 * @author Christian Skarby
 */
public class ParameterGradientView {

    private final Gradient allGradients;
    private final INDArray1DView realGradientView;

    public ParameterGradientView(Gradient allGradients, INDArray1DView realGradientView) {
        this.allGradients = allGradients;
        this.realGradientView = realGradientView;
    }


    /**
     * Returns all gradients (even those which are not actually gradients) per parameter
     * @return a Gradient for all parameters
     */
    public Gradient allGradientsPerParam() {
       return allGradients;
    }

    /**
     * Returns an {@link INDArray1DView} of only the parts of the gradient view which are actually gradients.
     * @return and {@link INDArray1DView}
     */
    public INDArray1DView realGradientView() {
        return realGradientView;
    }

}

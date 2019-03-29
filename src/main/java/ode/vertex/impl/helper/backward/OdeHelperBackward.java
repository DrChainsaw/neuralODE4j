package ode.vertex.impl.helper.backward;

import lombok.AllArgsConstructor;
import lombok.Getter;
import ode.vertex.impl.gradview.INDArray1DView;
import ode.vertex.impl.helper.GraphInputOutput;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Helps with input/output handling when solving ODEs inside a neural network
 *
 * @author Christian Skarby
 */
public interface OdeHelperBackward {

    /**
     * Input arrays needed to do backward pass. Has the following definitions:<br>
     * {@code lossGradient}: Gradient w.r.t loss from subsequent layers (typically called epsilon in dl4j)<br>
     * {@code lastOutput}: Last computed output from a forward pass used to calculate the loss gradient<br>
     * {@code realGradientView}: View of all array elements which are actually gradients in the given
     * {@link ComputationGraph}s gradient view array. Notable exceptions (i.e. things labeled as gradients which are not
     * are running mean and variance of Batch Normalization layers.
     */
    @Getter @AllArgsConstructor
    class InputArrays {

        private final GraphInputOutput graphInputOutput;
        private final INDArray lastOutput;
        private final INDArray lossGradient;
        private final INDArray1DView realGradientView;
    }

    /**
     * Misc parameters needed to jump through the hoops of doing back propagation
     */
    @Getter @AllArgsConstructor
    class MiscPar {
        private final boolean useTruncatedBackPropTroughTime;
        private final LayerWorkspaceMgr wsMgr;
    }

    /**
     * Return the solution to the ODE when assuming that a backwards pass through the layers of the given graph is
     * the derivative of the sought function. Note that parameter gradient is set in given graph so it is not returned.
     *
     * @param graph Graph of layers to do backwards pass through
     * @param input Input arrays
     * @param miscPars Misc parameters needed to jump through the hoops of doing back propagation
     *
     * @return Loss gradients (a.k.a epsilon in dl4j) w.r.t last input from previous layers. Note that parameter gradients
     * are set in graph and can be accessed through graph.getGradientsViewArray()
     */
    INDArray[] solve(ComputationGraph graph, InputArrays input, MiscPar miscPars);
}

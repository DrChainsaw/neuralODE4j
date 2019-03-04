package ode.vertex.impl.helper.backward;

import lombok.AllArgsConstructor;
import lombok.Data;
import ode.vertex.impl.NonContiguous1DView;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Helps with input/output handling when solving ODEs inside a neural network
 *
 * @author Christian Skarby
 */
public interface OdeHelperBackward {

    /**
     * Input arrays needed to do backward pass. Has the following definitions:<br>
     * {@code lossGradient}: Gradient w.r.t loss from subsequent layers (typically called epsilon in dl4j)<br>
     * {@code lossGradientTime}: Gradient w.r.t loss from subsequent layers w.r.t time<br>
     * {@code lastOutput}: Last computed output from a forward pass used to calculate the loss gradient<br>
     * {@code realGradientView}: View of all array elements which are actually gradients in the given
     * {@link ComputationGraph}s gradient view array. Notable exceptions (i.e. things labeled as gradients which are not
     * are running mean and variance of Batch Normalization layers.
     */
    @Data @AllArgsConstructor
    class InputArrays {

        public InputArrays(INDArray[] lastInputs, INDArray lastOutput, INDArray lossGradient, NonContiguous1DView realGradientView) {
            this(lastInputs, lastOutput, lossGradient, lossGradient, realGradientView);
        }

        private final INDArray[] lastInputs;
        private final INDArray lastOutput;
        private final INDArray lossGradient;
        private final INDArray lossGradientTime;
        private final NonContiguous1DView realGradientView;
    }

    /**
     * Misc parameters needed to jump through the hoops of doing back propagation
     */
    @Data
    class MiscPar {
        private final boolean useTruncatedBackPropTroughTime;
        private final LayerWorkspaceMgr wsMgr;
        private final String gradientParName;
    }

    /**
     * Return the solution to the ODE when assuming that a backwards pass through the layers of the given graph is
     * the derivative of the sought function.
     *
     * @param graph Graph of layers to do backwards pass through
     * @param input Input arrays
     * @param miscPars Misc parameters needed to jump through the hoops of doing back propagation
     *
     * @return Pair consisting of parameter {@link Gradient} for parameter adjustment and gradient w.r.t last input from previous layers.
     */
    Pair<Gradient, INDArray[]> solve(ComputationGraph graph, InputArrays input, MiscPar miscPars);
}

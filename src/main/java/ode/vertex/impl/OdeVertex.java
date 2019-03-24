package ode.vertex.impl;

import lombok.AllArgsConstructor;
import lombok.Getter;
import ode.vertex.impl.helper.OdeGraphHelper;
import ode.vertex.impl.helper.backward.OdeHelperBackward;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Implementation of an ODE block. Contains a {@link ComputationGraph} which defines the learnable function
 * {@code f = z(t)/dt} for which the {@code OdeVertex} will output an estimate of z(t) for given t(s).
 *
 * @author Christian Skarby
 */
public class OdeVertex extends BaseGraphVertex {

    private static final Logger log = LoggerFactory.getLogger(OdeVertex.class);

    private final OdeGraphHelper odeHelper;
    private final TrainingConfig trainingConfig;

    @AllArgsConstructor
    @Getter
    public static class BaseGraphVertexInputs {
        private final ComputationGraph graph;
        private final String name;
        private final int vertexIndex;
    }

    public OdeVertex(BaseGraphVertexInputs baseGraphVertexInputs,
                     OdeGraphHelper odeHelper,
                     TrainingConfig trainingConfig) {
        super(baseGraphVertexInputs.getGraph(), baseGraphVertexInputs.getName(), baseGraphVertexInputs.getVertexIndex(), null, null);
        this.trainingConfig = trainingConfig;
        this.odeHelper = odeHelper;
    }

    @Override
    public String toString() {
        return odeHelper.getFunction().toString();
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public long numParams() {
        return odeHelper.getFunction().numParams();
    }

    @Override
    public INDArray params() {
        return odeHelper.getFunction().params();
    }

    @Override
    public void clear() {
        super.clear();
        odeHelper.clear();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropOnly) {
        return odeHelper.paramTable(backpropOnly);
    }

    @Override
    public TrainingConfig getConfig() {
        return trainingConfig;
    }

    private void validateForward() {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");
    }

    private void validateBackprop() {
        if (!canDoBackward()) {
            if (inputs == null || inputs[0] == null) {
                throw new IllegalStateException("Cannot do backward pass: inputs not set. Layer: \"" + vertexName
                        + "\" (idx " + vertexIndex + "), numInputs: " + getNumInputArrays());
            } else {
                throw new IllegalStateException("Cannot do backward pass: all epsilons not set. Layer \"" + vertexName
                        + "\" (idx " + vertexIndex + "), numInputs :" + getNumInputArrays() + "; numOutputs: "
                        + getNumOutputConnections());
            }
        }
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        validateForward();

        log.trace("Start forward. Training: " + training);

        leverageInputs(workspaceMgr);

        final INDArray output = odeHelper.doForward(workspaceMgr, getInputs());

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, output);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();
        log.trace("Start backward");

        final Pair<Gradient, INDArray[]> gradients = odeHelper.doBackward(
                new OdeHelperBackward.MiscPar(tbptt, workspaceMgr),
                getEpsilon(),
                getInputs());

        final INDArray[] inputGrads = gradients.getSecond();
        final INDArray[] leveragedGrads = new INDArray[inputGrads.length];
        for (int i = 0; i < inputGrads.length; i++) {
            leveragedGrads[i] = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, inputGrads[i]);
        }

        return new Pair<>(gradients.getFirst(), leveragedGrads);
    }

    private void leverageInputs(LayerWorkspaceMgr workspaceMgr) {
        for (int i = 0; i < getInputs().length; i++) {
            setInput(i, workspaceMgr.leverageTo(ArrayType.INPUT, getInputs()[i]), workspaceMgr);
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        odeHelper.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}

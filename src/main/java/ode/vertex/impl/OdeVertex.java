package ode.vertex.impl;

import lombok.AllArgsConstructor;
import lombok.Getter;
import ode.vertex.impl.helper.backward.OdeHelperBackward;
import ode.vertex.impl.helper.forward.OdeHelperForward;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Implementation of an ODE block.
 *
 * @author Christian Skarby
 */
public class OdeVertex extends BaseGraphVertex {

    private static final Logger log = LoggerFactory.getLogger(OdeVertex.class);

    private final static String parName = "params";

    private final ComputationGraph graph;
    private final OdeHelperForward odeHelperForward;
    private final OdeHelperBackward odeHelperBackward;
    private final TrainingConfig trainingConfig;
    private final Parameters parameters;

    private static class Parameters {

        private INDArray lastOutput; // z(t1) from paper
        private final NonContiguous1DView realGradients; // Parts of graph.getFlattenedGradients() which are actually gradients

        private Parameters() {
            this.realGradients = new NonContiguous1DView();
        }

        private INDArray lastOutput() {
            return lastOutput;
        }

        private NonContiguous1DView realGradients() {
            return realGradients;
        }

        private void setLastOutput(INDArray lastOutput) {
            this.lastOutput = lastOutput;
        }
    }

    @AllArgsConstructor @Getter
    public static class BaseGraphVertexInputs {
        private final ComputationGraph graph;
        private final String name;
        private final int vertexIndex;
    }

    public OdeVertex(BaseGraphVertexInputs baseGraphVertexInputs,
                     ComputationGraph innerGraph,
                     OdeHelperForward odeHelperForward,
                     OdeHelperBackward odeHelperBackward,
                     TrainingConfig trainingConfig) {
        super(baseGraphVertexInputs.getGraph(), baseGraphVertexInputs.getName(), baseGraphVertexInputs.getVertexIndex(), null, null);
        this.graph = innerGraph;
        this.trainingConfig = trainingConfig;
        this.odeHelperForward = odeHelperForward;
        this.odeHelperBackward = odeHelperBackward;
        this.parameters = new Parameters();
    }

    @Override
    public String toString() {
        return graph.toString();
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
        return graph.numParams();
    }

    @Override
    public INDArray params() {
        return graph.params();
    }

    @Override
    public void clear() {
        super.clear();
        graph.clearLayersStates();
        parameters.setLastOutput(null);
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropOnly) {
        return Collections.synchronizedMap(Collections.singletonMap(parName, params()));
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

        final LayerWorkspaceMgr innerWorkspaceMgr = createWorkspaceMgr(workspaceMgr);

        graph.getConfiguration().setIterationCount(0);
        final INDArray output = odeHelperForward.solve(graph, innerWorkspaceMgr, getInputs());
        log.info("Nrof func eval forward " + graph.getIterationCount());

        parameters.setLastOutput(output.detach());
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, output);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();
        log.trace("Start backward");

        final OdeHelperBackward.InputArrays inputArrays = new OdeHelperBackward.InputArrays(
                getInputs(),
                parameters.lastOutput(),
                getEpsilon(),
                parameters.realGradients()
        );

        final OdeHelperBackward.MiscPar miscPar = new OdeHelperBackward.MiscPar(
                tbptt,
                createWorkspaceMgr(workspaceMgr),
                parName
        );

        graph.getConfiguration().setIterationCount(0);
        final Pair<Gradient, INDArray[]> gradients = odeHelperBackward.solve(graph, inputArrays, miscPar);
        log.info("Nrof func eval backward " + graph.getIterationCount());


        final INDArray[] inputGrads = gradients.getSecond();
        final INDArray[] leveragedGrads = new INDArray[inputGrads.length];
        for(int i = 0; i < inputGrads.length; i++) {
            leveragedGrads[i] = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, inputGrads[i]);
        }

        return new Pair<>(gradients.getFirst(), leveragedGrads);
    }

    private void leverageInputs(LayerWorkspaceMgr workspaceMgr) {
        for (int i = 0; i < getInputs().length; i++) {
            setInput(i, workspaceMgr.leverageTo(ArrayType.INPUT, getInputs()[i]), workspaceMgr);
        }
    }

    private LayerWorkspaceMgr createWorkspaceMgr(final LayerWorkspaceMgr outerWsMgr) {

        if(outerWsMgr == LayerWorkspaceMgr.noWorkspacesImmutable()) {
            return outerWsMgr;
        }

        return new ComputationGraph(graph.getConfiguration()) {
            LayerWorkspaceMgr spyWsConfigs() {
                // A little bit too many methods to comfortably decorate. Try to copy config instead
                final LayerWorkspaceMgr.Builder wsBuilder = LayerWorkspaceMgr.builder();
                for (ArrayType type : ArrayType.values()) {
                    if (outerWsMgr.hasConfiguration(type)) {
                        wsBuilder.with(type, outerWsMgr.getWorkspaceName(type), outerWsMgr.getConfiguration(type));
                    }
                }

                final LayerWorkspaceMgr wsMgr = wsBuilder
                        .with(ArrayType.FF_WORKING_MEM, "WS_ODE_VERTEX_LAYER_WORKING_MEM", WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.BP_WORKING_MEM, "WS_ODE_VERTEX_LAYER_WORKING_MEM", WS_LAYER_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, "WS_ODE_VERTEX_RNN_LOOP_WORKING_MEM", WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, "WS_ODE_VERTEX_RNN_LOOP_WORKING_MEM", WS_RNN_LOOP_WORKING_MEM_CONFIG)
                        .with(ArrayType.ACTIVATIONS, "WS_ODE_VERTEX_ALL_LAYERS_ACT", WS_ALL_LAYERS_ACT_CONFIG)
                        .with(ArrayType.ACTIVATION_GRAD, "WS_ODE_VERTEX_ALL_LAYERS_GRAD", WS_ALL_LAYERS_ACT_CONFIG)
                        .build();
                wsMgr.setHelperWorkspacePointers(outerWsMgr.getHelperWorkspacePointers());
                return wsMgr;
            }
        }.spyWsConfigs();

    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        graph.setBackpropGradientsViewArray(backpropGradientsViewArray);

        // What is this about? Some layers "abuse" the gradient to perform updates of parameters for which no gradient
        // is calculated and this screws up the ODE solvers idea of what the solution is. The following layers are known
        // to do this:
        //
        // * BatchNormalization: The global variance and mean are just the (sliding) average of the batch dittos.
        //                       However, in order to support distributed training the updates are performed by adding
        //                       the change to the state as a gradient even through it is not really.

        // Maybe get these from config so user can specify others e.g. for custom layers
        final List<String> nonGradientParamNames = Arrays.asList(
                BatchNormalizationParamInitializer.GLOBAL_VAR,
                BatchNormalizationParamInitializer.GLOBAL_MEAN);

        parameters.realGradients().clear();
        for (Layer layer : graph.getLayers()) {
            Map<String, INDArray> gradParams = layer.conf().getLayer().initializer().getGradientsFromFlattened(layer.conf(), layer.getGradientsViewArray());
            for (Map.Entry<String, INDArray> parNameAndGradView : gradParams.entrySet()) {

                final String parName = parNameAndGradView.getKey();
                final INDArray grad = parNameAndGradView.getValue();

                if (!nonGradientParamNames.contains(parName)) {
                    parameters.realGradients().addView(grad.reshape(grad.length()));
                }
            }
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}

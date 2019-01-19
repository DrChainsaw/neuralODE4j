package ode.vertex.impl;

import com.google.common.primitives.Longs;
import lombok.AllArgsConstructor;
import lombok.Getter;
import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.impl.MultiStepSolver;
import ode.vertex.impl.helper.NDArrayIndexAccumulator;
import ode.vertex.impl.helper.backward.AugmentedDynamics;
import ode.vertex.impl.helper.backward.BackpropagateAdjoint;
import ode.vertex.impl.helper.forward.ForwardPass;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Implementation of an ODE block.
 *
 * @author Christian Skarby
 */
public class OdeVertex extends BaseGraphVertex {

    private static final Logger log = LoggerFactory.getLogger(OdeVertex.class);

    private final static String parName = "params";

    private final ComputationGraph graph;
    private final FirstOrderSolver odeSolver;
    private final TrainingConfig trainingConfig;
    private final Parameters parameters;

    private abstract class Parameters {

        private INDArray lastOutput; // z(t1) from paper
        private final NonContiguous1DView realGradients; // Parts of graph.getFlattenedGradients() which are actually gradients

        private Parameters() {
            this.realGradients = new NonContiguous1DView();
        }

        abstract INDArray time();

        abstract INDArray[] inputs();

        abstract INDArray[] epsilons(LayerWorkspaceMgr wsMgr, AugmentedDynamics finalState);

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

    private class ParametersFixedTime extends Parameters {
        private final INDArray time;

        private ParametersFixedTime(INDArray time) {
            super();
            this.time = time;
        }

        @Override
        INDArray time() {
            return time;
        }

        @Override
        INDArray[] inputs() {
            return getInputs();
        }

        @Override
        INDArray[] epsilons(LayerWorkspaceMgr wsMgr, AugmentedDynamics finalState) {
            return new INDArray[] {wsMgr.leverageTo(ArrayType.ACTIVATION_GRAD, finalState.zAdjoint())};
        }
    }

    private class ParametersTimeAsInput extends Parameters {

        private final int timeInputIndex;

        private ParametersTimeAsInput(int timeInputIndex) {
            super();
            this.timeInputIndex = timeInputIndex;
        }

        @Override
        INDArray time() {
            return getInputs()[timeInputIndex];
        }

        @Override
        INDArray[] inputs() {
            final List<INDArray> notTimeInputs = new ArrayList<>();
            for(int i = 0; i < getInputs().length; i++) {
                if(i != timeInputIndex) {
                    notTimeInputs.add(getInputs()[i]);
                }
            }
            return notTimeInputs.toArray(new INDArray[0]);
        }

        @Override
        INDArray[] epsilons(LayerWorkspaceMgr wsMgr, AugmentedDynamics finalState) {
            final INDArray[] epsilons = new INDArray[getInputs().length];
            for(int i = 0; i < getInputs().length; i++) {
                if(i != timeInputIndex) {
                    epsilons[i] = wsMgr.leverageTo(ArrayType.ACTIVATION_GRAD, finalState.zAdjoint());
                } else {
                    epsilons[i] = wsMgr.leverageTo(ArrayType.ACTIVATION_GRAD, finalState.tAdjoint());
                }
            }
            return epsilons;
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
                     FirstOrderSolver odeSolver,
                     TrainingConfig trainingConfig,
                     int timeAsInputIndex) {
        super(baseGraphVertexInputs.getGraph(), baseGraphVertexInputs.getName(), baseGraphVertexInputs.getVertexIndex(), null, null);
        this.graph = innerGraph;
        this.trainingConfig = trainingConfig;
        this.odeSolver = odeSolver;
        this.parameters = timeAsInputIndex != -1 ? new ParametersTimeAsInput(timeAsInputIndex) : new ParametersFixedTime(Nd4j.create(new double[]{0, 1}));
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

        if (parameters.inputs().length != 1) {
            throw new IllegalStateException("Only one input supported!");
        }
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

        final ForwardPass equation = new ForwardPass(
                graph,
                innerWorkspaceMgr,
                true, // Always use training as batch norm running mean and var become messed up otherwise. Same effect seen in original pytorch repo.
                parameters.inputs());

        // nrof outputs must be same as number of inputs due to resblock
        final INDArray output = addTimeStepsToOutput(Nd4j.createUninitialized(parameters.inputs()[0].shape()));
        new MultiStepSolver(odeSolver).integrate(equation, parameters.time(), parameters.inputs()[0], output);

        for (GraphVertex vertex : graph.getVertices()) {
            final INDArray[] inputs = vertex.getInputs();
            for (int i = 0; i < inputs.length; i++) {
                vertex.setInput(i, workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[i]), workspaceMgr);
            }
        }

        parameters.setLastOutput(output.detach());
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, alignOutShape(output));
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();
        log.trace("Start backward");

        // Create augmented dynamics for adjoint method
        // Initialization: S0:
        // z(t1) = lastoutput
        // a(t) = -dL/d(z(t1)) = -epsilon from next layer (i.e getEpsilon). Use last row if more than one timestep
        // parameters = zeros
        // dL/dt1 = dL / dz(t1) dot z(t1)

        // TODO: This is just a placeholder to verify that dims work out! Need to do all the below for each discrete timestep and not
        //  use MultiStepSolver
        final INDArray dL_dztN = getLast(alignInShape(getEpsilon()));
        final INDArray ztN = getLast(parameters.lastOutput());

        final INDArray dL_dtN = dL_dztN.reshape(1, ztN.length())
                .mmul(ztN.reshape(ztN.length(), 1)).muli(-1);

        final INDArray zAug = Nd4j.create(1, ztN.length() + dL_dztN.length() + graph.numParams() + dL_dtN.length());

        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(ztN.reshape(new long[]{1, ztN.length()}))
                .increment(dL_dztN.reshape(new long[]{1, dL_dztN.length()}))
                .increment(Nd4j.zeros(parameters.realGradients().length()).reshape(new long[]{1, Nd4j.zeros(parameters.realGradients().length()).length()}))
                .increment(dL_dtN.reshape(new long[]{1, dL_dtN.length()}));

        final AugmentedDynamics augmentedDynamics = new AugmentedDynamics(
                zAug,
                dL_dztN.shape(),
                new long[]{parameters.realGradients().length()},
                dL_dtN.shape());

        final LayerWorkspaceMgr innerWorkspaceMgr = createWorkspaceMgr(workspaceMgr);

        final FirstOrderEquation equation = new BackpropagateAdjoint(
                augmentedDynamics,
                new ForwardPass(graph,
                        innerWorkspaceMgr,
                        true,
                        parameters.inputs()),
                new BackpropagateAdjoint.GraphInfo(graph, parameters.realGradients(), innerWorkspaceMgr, tbptt)
        );

        INDArray augAns = new MultiStepSolver(odeSolver).integrate(equation, Nd4j.reverse(parameters.time().dup()), zAug, addTimeStepsToOutput(zAug.dup()));

        augmentedDynamics.updateFrom(getLast(augAns));

        parameters.realGradients().assignFrom(augmentedDynamics.paramAdjoint());
        final Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        gradient.setGradientFor(parName, graph.getFlattenedGradients());

        return new Pair<>(gradient, parameters.epsilons(workspaceMgr, augmentedDynamics));
    }

    private void leverageInputs(LayerWorkspaceMgr workspaceMgr) {
        for (int i = 0; i < getInputs().length; i++) {
            setInput(i, workspaceMgr.leverageTo(ArrayType.INPUT, getInputs()[i]), workspaceMgr);
        }
    }

    private INDArray addTimeStepsToOutput(INDArray output) {
        if(parameters.time().length() == 2) {
            return output.reshape(Longs.concat(new long[] {1}, output.shape()));
        }

        return Nd4j.repeat(output, (int)parameters.time().length()-1);
    }

    private INDArray alignOutShape(INDArray array) {
        if(parameters.time().length() == 2) {
            return array.reshape(getInputs()[0].shape());
        }

        final long[] shape = array.shape();
        switch (shape.length) {
            case 3: // Assume recurrent output
                return Nd4j.concat(0, getInputs()[0].reshape(1, shape[1], shape[2]), array).permute(1,2,0);
            case 5: // Assume conv 3D output
                return Nd4j.concat(0, getInputs()[0].reshape(1, shape[1], shape[2], shape[3], shape[4]), array).permute(1,0,2,3,4);
                // Should not happen as conf throws exception for other types
                default: throw new UnsupportedOperationException("Rank not supported: " + array.rank());
        }
    }

    private INDArray alignInShape(INDArray array) {
        if(parameters.time().length() == 2) {
            return array.reshape(getInputs()[0].shape());
        }

        final long[] shape = array.shape();
        switch (shape.length) {
            case 3: // Assume recurrent output
                return array.permute(2,0,1);
            case 5: // Assume conv 3D output
                return array.permute(1,0,2,3,4);
            // Should not happen as conf throws exception for other types
            default: throw new UnsupportedOperationException("Rank not supported: " + array.rank());
        }
    }

    private INDArray getLast(INDArray array) {
        if(parameters.time().length() == 2) {
            return array;
        }

        final INDArrayIndex[] last = new INDArrayIndex[array.rank()];
        for(int i = 1; i < last.length; i++) {
            last[i] = NDArrayIndex.all();
        }
        last[0] = NDArrayIndex.point(array.size(0)-1);
        return array.get(last);
    }

    private LayerWorkspaceMgr createWorkspaceMgr(final LayerWorkspaceMgr outerWsMgr) {

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

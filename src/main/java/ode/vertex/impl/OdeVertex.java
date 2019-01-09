package ode.vertex.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
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
    private final FirstOrderSolver odeSolver;
    private final TrainingConfig trainingConfig;
    private final Parameters parameters;

    private static class Parameters {
        private final INDArray time;
        private INDArray lastOutput; // z(t1) from paper
        private final NonContiguous1DView realGradients; // Parts of graph.getFlattenedGradients() which are actually gradients

        public Parameters(INDArray time) {
            this.time = time;
            realGradients = new NonContiguous1DView();
        }

    }

    public OdeVertex(ComputationGraph actualGraph,
                     String name,
                     int vertexIndex,
                     ComputationGraph innerGraph,
                     FirstOrderSolver odeSolver,
                     TrainingConfig trainingConfig) {
        super(actualGraph, name, vertexIndex, null, null);
        this.graph = innerGraph;
        this.trainingConfig = trainingConfig;
        this.odeSolver = odeSolver;
        this.parameters = new Parameters(Nd4j.create(new double[]{0, 1}));
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
        parameters.lastOutput = null;
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

        if (getInputs().length != 1) {
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
                getInputs());

        // nrof outputs must be same as number of inputs due to resblock
        parameters.lastOutput = workspaceMgr.createUninitialized(ArrayType.INPUT, getInputs()[0].shape()).detach();
        odeSolver.integrate(equation, parameters.time, getInputs()[0], parameters.lastOutput);

        for (GraphVertex vertex : graph.getVertices()) {
            final INDArray[] inputs = vertex.getInputs();
            for (int i = 0; i < inputs.length; i++) {
                vertex.setInput(i, workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[i]), workspaceMgr);
            }
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, parameters.lastOutput);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();
        log.trace("Start backward");

        // Create augmented dynamics for adjoint method
        // Initialization: S0:
        // z(t1) = lastoutput
        // a(t) = -dL/d(z(t1)) = -epsilon from next layer (i.e getEpsilon)
        // parameters = zeros
        // dL/dt1 = dL / dz(t1) dot z(t1)
        final INDArray dL_dtN = getEpsilon().reshape(1, parameters.lastOutput.length())
                .mmul(parameters.lastOutput.reshape(parameters.lastOutput.length(), 1)).muli(-1);

        final INDArray zAug = Nd4j.create(1, parameters.lastOutput.length() + getEpsilon().length() + graph.numParams() + dL_dtN.length());

        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(zAug);
        accumulator.increment(parameters.lastOutput.reshape(new long[]{1, parameters.lastOutput.length()}))
                .increment(getEpsilon().reshape(new long[]{1, getEpsilon().length()}))
                .increment(Nd4j.zeros(parameters.realGradients.length()).reshape(new long[]{1, Nd4j.zeros(parameters.realGradients.length()).length()}))
                .increment(dL_dtN.reshape(new long[]{1, dL_dtN.length()}));

        final AugmentedDynamics augmentedDynamics = new AugmentedDynamics(
                zAug,
                getEpsilon().shape(),
                new long[]{parameters.realGradients.length()},
                dL_dtN.shape());

        final LayerWorkspaceMgr innerWorkspaceMgr = createWorkspaceMgr(workspaceMgr);

        final FirstOrderEquation equation = new BackpropagateAdjoint(
                augmentedDynamics,
                new ForwardPass(graph,
                        innerWorkspaceMgr,
                        true,
                        getInputs()),
                new BackpropagateAdjoint.GraphInfo(graph, parameters.realGradients, innerWorkspaceMgr, tbptt)
        );

        INDArray augAns = odeSolver.integrate(equation, Nd4j.reverse(parameters.time.dup()), zAug, zAug.dup());

        ((BackpropagateAdjoint) equation).updatePreTimer.logSum("update state pre");
        ((BackpropagateAdjoint) equation).forward.logSum("forward");
        ((BackpropagateAdjoint) equation).gradTimer.logSum("grad calc");
        ((BackpropagateAdjoint) equation).updatePostTimer.logSum("update state post");
        ((BackpropagateAdjoint) equation).assignTo.logSum("assign to from non cont view");
        System.out.println("Nfe backwards: " + ((BackpropagateAdjoint) equation).nfe);

        augmentedDynamics.updateFrom(augAns);

        final INDArray epsilonOut = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, augmentedDynamics.zAdjoint());

        parameters.realGradients.assignFrom(augmentedDynamics.paramAdjoint());
        final Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        gradient.setGradientFor(parName, graph.getFlattenedGradients());

        return new Pair<>(gradient, new INDArray[]{epsilonOut});
    }

    private void leverageInputs(LayerWorkspaceMgr workspaceMgr) {
        for (int i = 0; i < getInputs().length; i++) {
            setInput(i, workspaceMgr.leverageTo(ArrayType.INPUT, getInputs()[i]), workspaceMgr);
        }
    }

    private LayerWorkspaceMgr createWorkspaceMgr(final LayerWorkspaceMgr outerWsMgr) {

        return new ComputationGraph(graph.getConfiguration()) {
            public LayerWorkspaceMgr spyWsConfigs() {
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

        parameters.realGradients.clear();
        for (Layer layer : graph.getLayers()) {
            Map<String, INDArray> gradParams = layer.conf().getLayer().initializer().getGradientsFromFlattened(layer.conf(), layer.getGradientsViewArray());
            for (Map.Entry<String, INDArray> parNameAndGradView : gradParams.entrySet()) {

                final String parName = parNameAndGradView.getKey();
                final INDArray grad = parNameAndGradView.getValue();

                if (!nonGradientParamNames.contains(parName)) {
                    parameters.realGradients.addView(grad.reshape(grad.length()));
                }
            }
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}

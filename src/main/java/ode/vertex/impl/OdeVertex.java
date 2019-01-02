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
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
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
    private final INDArray time;
    private INDArray lastOutput; // z(t1) from paper

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
        time = Nd4j.create(new double[]{0, 1});
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
        lastOutput = null;
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
        // Not sure if needed anymore...
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {

            final LayerWorkspaceMgr innerWorkspaceMgr = createWorkspaceMgr(workspaceMgr);

            final ForwardPass equation = new ForwardPass(
                    graph,
                    innerWorkspaceMgr,
                    training,
                    getInputs());

            lastOutput = Nd4j.createUninitialized(getInputs()[0].shape()).detach(); // nrof outputs must be same as number of inputs due to resblock
            odeSolver.integrate(equation, time, getInputs()[0], lastOutput);
        }

        for (GraphVertex vertex : graph.getVertices()) {
            final INDArray[] inputs = vertex.getInputs();
            for (int i = 0; i < inputs.length; i++) {
                vertex.setInput(i, workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[i]), workspaceMgr);
            }
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, lastOutput);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();
        log.trace("Start backward");

        final AugmentedDynamics augmentedDynamics;
        // Not sure if needed anymore...
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {

            // Create augmented dynamics for adjoint method
            // Initialization: S0:
            // z(t1) = lastoutput
            // a(t) = -dL/d(z(t1)) = -epsilon from next layer (i.e getEpsilon)
            // parameters = zeros
            // dL/dt1 = dL / dz(t1) dot z(t1)
            final INDArray dL_dtN = getEpsilon().reshape(1, lastOutput.length())
                    .mmul(lastOutput.reshape( lastOutput.length(),1)).muli(-1);

            augmentedDynamics = new AugmentedDynamics(
                    lastOutput.dup(),
                    getEpsilon().dup(),
                    Nd4j.zeros(graph.params().shape()),
                    dL_dtN);

            final LayerWorkspaceMgr innerWorkspaceMgr = createWorkspaceMgr(workspaceMgr);

            final FirstOrderEquation equation = new BackpropagateAdjoint(
                    graph,
                    innerWorkspaceMgr,
                    augmentedDynamics,
                    new ForwardPass(graph,
                            innerWorkspaceMgr,
                            true,
                            getInputs()),
                    tbptt
            );

            final INDArray zAug = Nd4j.create(1, lastOutput.length() + getEpsilon().length() + graph.numParams() + dL_dtN.length());
            augmentedDynamics.transferTo(zAug);

            INDArray augAns = odeSolver.integrate(equation, Nd4j.reverse(time.dup()), zAug, zAug.dup());

            augmentedDynamics.updateFrom(augAns);
        }

        final INDArray epsilonOut = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, augmentedDynamics.getZAdjoint());

        graph.getFlattenedGradients().assign(augmentedDynamics.getParamAdjoint());
        final Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        gradient.setGradientFor(parName, graph.getFlattenedGradients());

        return new Pair<>(gradient, new INDArray[]{epsilonOut});
    }

    private void leverageInputs(LayerWorkspaceMgr workspaceMgr) {
        for(int i = 0; i < getInputs().length; i++) {
            setInput(i, workspaceMgr.leverageTo(ArrayType.INPUT, getInputs()[i]), workspaceMgr);
        }
    }

    private LayerWorkspaceMgr createWorkspaceMgr(final LayerWorkspaceMgr outerWsMgr) {

         return new ComputationGraph(graph.getConfiguration()) {
             public LayerWorkspaceMgr spyWsConfigs() {
                 // A little bit too many methods to comfortablty decorate. Try to copy config instead
                 final LayerWorkspaceMgr.Builder wsBuilder = LayerWorkspaceMgr.builder();
                 for(ArrayType type: ArrayType.values()) {
                     if(outerWsMgr.hasConfiguration(type)) {
                         wsBuilder.with(type, outerWsMgr.getWorkspaceName(type), outerWsMgr.getConfiguration(type));
                     }
                 }

                 final LayerWorkspaceMgr wsMgr =  wsBuilder
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
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}

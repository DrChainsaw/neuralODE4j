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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Map;

/**
 * Implementation of an ODE block.
 *
 * @author Christian Skarby
 */
public class OdeVertex extends BaseGraphVertex {

    private final static String parName = "params";

    private final ComputationGraph graph;
    private final FirstOrderSolver odeSolver;
    private final TrainingConfig trainingConfig;
    private final INDArray time;
    private INDArray lastOutput; // z(tN) from paper?
    private final LayerWorkspaceMgr innerWorkspaceMgr;

    public OdeVertex(ComputationGraph actualGraph,
                     String name,
                     int vertexIndex,
                     ComputationGraph innerGraph,
                     FirstOrderSolver odeSolver,
                     TrainingConfig trainingConfig,
                     LayerWorkspaceMgr innerWorkspaceMgr) {
        super(actualGraph, name, vertexIndex, null, null);
        this.graph = innerGraph;
        this.trainingConfig = trainingConfig;
        this.odeSolver = odeSolver;
        time = Nd4j.create(new double[]{0, 1});
        this.innerWorkspaceMgr = innerWorkspaceMgr;
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
        System.out.println("do forward!");
        final ForwardPass equation = new ForwardPass(graph, this.innerWorkspaceMgr, training, getInputs());
        lastOutput = Nd4j.createUninitialized(getInputs()[0].shape()).detach(); // nrof outputs must be same as number of inputs due to resblock

        odeSolver.integrate(equation, time, getInputs()[0], lastOutput);

        for(GraphVertex vertex: graph.getVertices()) {
            final INDArray[] inputs = vertex.getInputs();
            for(int i = 0; i < inputs.length; i++) {
                vertex.setInput(i, workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, inputs[i]), workspaceMgr);
            }
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, lastOutput);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();
        System.out.println("do backward!");
        // epsilon = dL / dz(tN) = dL / dlastOutput
        // dL/dtN = dL / dz(tN) dot z(tN)
        final INDArray dL_dtN = Nd4j.toFlattened(getEpsilon()).mmul(Nd4j.toFlattened(lastOutput).transposei()).muli(-1);

        final AugmentedDynamics augmentedDynamics = new AugmentedDynamics(
                lastOutput.dup(),
                getEpsilon().dup(),
                Nd4j.zeros(graph.params().shape()),
                dL_dtN);

        final FirstOrderEquation equation = new BackpropagateAdjoint(
                graph,
                this.innerWorkspaceMgr,
                augmentedDynamics,
                new ForwardPass(graph,
                        this.innerWorkspaceMgr,
                        true,
                        getInputs()),
                tbptt
        );

        final INDArray zAug = Nd4j.create(1, lastOutput.length() + getEpsilon().length() + graph.numParams() + dL_dtN.length());
        augmentedDynamics.transferTo(zAug);

        INDArray augAns = odeSolver.integrate(equation, Nd4j.reverse(time.dup()), zAug, zAug.dup());

        augmentedDynamics.updateFrom(augAns);

        final INDArray epsilonOut = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, augmentedDynamics.getZAdjoint());

        graph.getFlattenedGradients().assign(augmentedDynamics.getParamAdjoint());
        final Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        gradient.setGradientFor(parName, graph.getFlattenedGradients());

        return new Pair<>(gradient, new INDArray[] {epsilonOut});
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

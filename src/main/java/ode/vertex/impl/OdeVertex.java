package ode.vertex.impl;

import ode.solve.api.FirstOrderEquation;
import ode.solve.api.FirstOrderSolver;
import ode.solve.commons.FirstOrderSolverAdapter;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
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
    private final LayerWorkspaceMgr workspaceMgr;

    public OdeVertex(ComputationGraph actualGraph,
                     String name,
                     int vertexIndex,
                     ComputationGraph innerGraph,
                     TrainingConfig trainingConfig,
                     LayerWorkspaceMgr workspaceMgr) {
        super(actualGraph, name, vertexIndex, null, null);
        this.graph = innerGraph;
        this.trainingConfig = trainingConfig;
        this.odeSolver = new FirstOrderSolverAdapter(
                new DormandPrince54Integrator(1e-10, 10d, 1e-2, 1e-2));
        time = Nd4j.create(new double[]{0, 1});
        this.workspaceMgr = workspaceMgr;
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
            throw new IllegalStateException("More than one input not supported!");
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

        final ForwardPass equation = new ForwardPass(graph, this.workspaceMgr, training, getInputs());
        final INDArray flatInputs = Nd4j.create(1, getNrofInputElements());
        lastOutput = Nd4j.create(1, getNrofInputElements()).detach(); // nrof outputs must be same as number of inputs due to resblock
        final NDArrayIndexAccumulator accumulator = new NDArrayIndexAccumulator(flatInputs);
        for (INDArray input : getInputs()) {
            accumulator.increment(Nd4j.toFlattened(input));
        }

        odeSolver.integrate(equation, time, flatInputs, lastOutput);

        if (equation.getLastOutputs().size() > 1) {
            throw new UnsupportedOperationException("More than one output not supported!");
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, equation.getLastOutputs().get(0));
    }

    private int getNrofInputElements() {
        int dimcnt = 0;
        for (INDArray input : getInputs()) {
            dimcnt += input.length();
        }
        return dimcnt;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        validateBackprop();

        // epsilon = dL / dz(tN) = dL / dlastOutput
        // dL/dtN = dL / dz(tN) dot z(tN)
        final INDArray dL_dtN = Nd4j.toFlattened(getEpsilon()).mmul(Nd4j.toFlattened(lastOutput).transposei()).muli(-1);

        final INDArray zAug = Nd4j.create(1, lastOutput.length() + getEpsilon().length() + graph.numParams() + dL_dtN.length());
        new NDArrayIndexAccumulator(zAug)
                .increment(lastOutput)
                .increment(Nd4j.toFlattened(getEpsilon()))
                .increment(Nd4j.zeros(1, graph.numParams()))
                .increment(dL_dtN);

        final AugmentedDynamics augmentedDynamics = new AugmentedDynamics(
                zAug,
                lastOutput.length(),
                graph.numParams(),
                dL_dtN.length(),
                getEpsilon().shape());

        final FirstOrderEquation equation = new BackpropagateAdjoint(
                graph,
                this.workspaceMgr,
                augmentedDynamics,
                new ForwardPass(graph,
                        this.workspaceMgr,
                        false,
                        getInputs()),
                tbptt
        );

        odeSolver.integrate(equation, Nd4j.reverse(time), zAug, zAug.dup());

        if (augmentedDynamics.getLastEpsilons().length > 1) {
            throw new UnsupportedOperationException("More the one input not supported!!");
        }

        for (INDArray eps : augmentedDynamics.getLastEpsilons()) {
            workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, eps.addi(getEpsilon()));
        }
        final Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        gradient.setGradientFor(parName, graph.getFlattenedGradients());
        return new Pair<>(gradient, augmentedDynamics.getLastEpsilons());
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

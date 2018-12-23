package ode.impl;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.misc.DummyConfig;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Implementation of an ODE block.
 *
 * @author Christian Skarby
 */
public class OdeVertex extends BaseGraphVertex {

    private final ComputationGraph graph;
    private final FirstOrderIntegrator odeSolver;
    private final INDArray time;
    private INDArray lastOutput; // z(tN) from paper?

    public OdeVertex(ComputationGraph actualGraph,
                     String name,
                     int vertexIndex,
                     VertexIndices[] inputVertices,
                     VertexIndices[] outputVertices,
                     ComputationGraph innerGraph) {
        super(actualGraph, name, vertexIndex, inputVertices, outputVertices);
        this.graph = innerGraph;
        odeSolver = new DormandPrince54Integrator(1e-8, 10d, 1e-9, 1e-7);
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
    public TrainingConfig getConfig() {
        return new DummyConfig(getVertexName());
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        if (getInputs().length != 1) {
            throw new IllegalStateException("More than one input not supported!");
        }

        final int dim = getNrofInputElements();
        final long[] shape = getInputs()[0].shape();

        final FirstOrderDifferentialEquations equation = new FirstOrderDifferentialEquations() {

            boolean first = true;

            @Override
            public int getDimension() {
                return dim;
            }

            @Override
            public void computeDerivatives(double t, double[] y, double[] yDot) throws MaxCountExceededException, DimensionMismatchException {
                if (!first) {
                    setInputsFromFlat(fromDoubleVec(y, workspaceMgr), workspaceMgr);
                }
                first = false;
                final INDArray eval = Nd4j.toFlattened(evaluate(training, workspaceMgr));
                System.arraycopy(eval.toDoubleVector(), 0, yDot, 0, yDot.length);
            }
        };

        final double[] output = new double[dim];
        odeSolver.integrate(equation, time.getDouble(0), output, time.getDouble(1), output);

        lastOutput = fromDoubleVec(output, shape, workspaceMgr);
        return lastOutput;
    }

    private int getNrofInputElements() {
        int dimcnt = 0;
        for (INDArray input : getInputs()) {
            dimcnt += input.length();
        }
        return dimcnt;
    }

    private INDArray fromDoubleVec(double[] vec, long[] shape, LayerWorkspaceMgr workspaceMgr) {
        return fromDoubleVec(vec, workspaceMgr).reshape(shape);
    }

    private INDArray fromDoubleVec(double[] vec, LayerWorkspaceMgr workspaceMgr) {
        final INDArray input = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, 1, vec.length);
        for (int i = 0; i < vec.length; i++) {
            input.putScalar(i, vec[i]);
        }
        return input;
    }

    private void setInputsFromFlat(INDArray flatArray, LayerWorkspaceMgr workspaceMgr) {
        int lastInd = 0;
        for (int i = 0; i < getInputs().length; i++) {
            INDArray input = getInputs()[i];
            final INDArray z = flatArray.get(NDArrayIndex.interval(lastInd, lastInd + input.length()));
            lastInd += input.length();
            setInput(i, z.reshape(input.shape()), workspaceMgr);
        }
    }

    @Nullable
    private INDArray evaluate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        //TODO: Might want to have internal workspaceMgr handling to conserve memory
        INDArray output = null;
        final int[] topologicalOrder = graph.topologicalSortOrder();

        //Do forward pass according to the topological ordering of the network
        for (int i = 0; i <= graph.getVertices().length - 1; i++) {
            GraphVertex current = graph.getVertices()[topologicalOrder[i]];
            String vName = current.getVertexName();
            int vIdx = current.getVertexIndex();

            VertexIndices[] inputsTo = current.getOutputVertices();

            INDArray out = null;
            if (current.isInputVertex()) {
                out = getInputs()[vIdx];
            } else if (current.isOutputVertex()) {
                output = current.getInputs()[0];
            } else {
                //Standard feed-forward case
                out = current.doForward(training, workspaceMgr);
            }

            if (inputsTo != null) {  //Output vertices may not input to any other vertices
                for (VertexIndices v : inputsTo) {
                    //Note that we don't have to do anything special here: the activations are always detached in
                    // this method
                    int inputToIndex = v.getVertexIndex();
                    int vIdxEdge = v.getVertexEdgeNumber();
                    graph.getVertices()[inputToIndex].setInput(vIdxEdge, out, workspaceMgr);
                }
            }

            //current.clear();

        }
        setInput(0, output, workspaceMgr);
        return output;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {

        // epsilon = dL / dz(tN) = dL / dlastOutput
        // dL/dtN = dL / dz(tN) dot z(tN)
        final INDArray dL_dtN = Nd4j.toFlattened(getEpsilon()).mmul(Nd4j.toFlattened(lastOutput).transposei()).muli(-1);

        final AugmentedDynamics augmentedDynamics = new AugmentedDynamics(lastOutput, getEpsilon(), Nd4j.zeros(graph.numParams()), dL_dtN);

        return backPropagate(getEpsilon(), tbptt, workspaceMgr);
    }

    private void updateAugmentedDynamics(AugmentedDynamics augmentedDynamics, INDArray time, LayerWorkspaceMgr workspaceMgr) {

        // Set inputs before eval
        setInputsFromFlat(augmentedDynamics.z, workspaceMgr);
        final INDArray fEval = evaluate(true, workspaceMgr);

    }

    private static class AugmentedDynamics {

        private final INDArray z;
        private final INDArray zAdjoint;
        private final INDArray paramAdjoint;
        private final INDArray tAdjoint;

        private AugmentedDynamics(double[] zAug, long nrofZ, long nrofParam, long nrofT) {
            this(Nd4j.create(zAug), nrofZ, nrofT, nrofParam);
        }

        private AugmentedDynamics(INDArray zAug, long nrofZ, long nrofParam, long nrofT) {
            this(
                    zAug.get(NDArrayIndex.interval(0, nrofZ)),
                    zAug.get(NDArrayIndex.interval(nrofZ, 2 * nrofZ)),
                    zAug.get(NDArrayIndex.interval(2 * nrofZ, 2 * nrofZ + nrofParam)),
                    zAug.get(NDArrayIndex.interval(2 * nrofZ + nrofParam, 2 * nrofZ + nrofParam + nrofT)));
        }

        private AugmentedDynamics(INDArray z, INDArray zAdjoint, INDArray paramAdjoint, INDArray tAdjoint) {
            this.z = z;
            this.zAdjoint = zAdjoint;
            this.paramAdjoint = paramAdjoint;
            this.tAdjoint = tAdjoint;
        }

        private void update(INDArray fEval, INDArray t) {

        }
    }

    private Pair<Gradient, INDArray[]> backPropagate(INDArray epsilon, boolean truncatedBPTT, LayerWorkspaceMgr workspaceMgr) {
        //Do backprop, in reverse topological order
        final int[] topologicalOrder = graph.topologicalSortOrder();
        final GraphVertex[] vertices = graph.getVertices();

        List<INDArray> outputEpsilons = new ArrayList<>();

        LinkedList<Triple<String, INDArray, Character>> gradients = new LinkedList<>();
        boolean[] setVertexEpsilon = new boolean[topologicalOrder.length]; //If true: already set epsilon for this vertex; later epsilons should be *added* to the existing one, not set
        for (int i = topologicalOrder.length - 1; i >= 0; i--) {
            GraphVertex current = vertices[topologicalOrder[i]];
            int vIdx = current.getVertexIndex();
            String vertexName = current.getVertexName();

            if (current.isOutputVertex()) {
                for(VertexIndices vertexIndices: current.getInputVertices()) {
                    final String inputName = vertices[vertexIndices.getVertexIndex()].getVertexName();
                    graph.getVertex(inputName).setEpsilon(epsilon);
                }
                continue;
            }

            if(current.isInputVertex()) {
                continue;
            }

            Pair<Gradient, INDArray[]> pair;
            INDArray[] epsilons;
            pair = current.doBackward(truncatedBPTT, workspaceMgr);
            epsilons = pair.getSecond();

            for(VertexIndices vertexIndices: current.getInputVertices()) {
                final String inputName = vertices[vertexIndices.getVertexIndex()].getVertexName();
                if(graph.getConfiguration().getNetworkInputs().contains(
                        inputName)) {
                    outputEpsilons.add(graph.getConfiguration().getNetworkInputs().indexOf(inputName),
                            epsilons[vertexIndices.getVertexEdgeNumber()]);
                }
            }

            //Inputs to the current GraphVertex:
            VertexIndices[] inputVertices = current.getInputVertices();

            //Set epsilons for the vertices that provide inputs to this vertex:
            if (inputVertices != null) {
                int j = 0;
                for (VertexIndices v : inputVertices) {
                    GraphVertex gv = graph.getVertices()[v.getVertexIndex()];
                    if (setVertexEpsilon[gv.getVertexIndex()]) {
                        //This vertex: must output to multiple vertices... we want to add the epsilons here
                        INDArray currentEps = gv.getEpsilon();
                        gv.setEpsilon(currentEps.addi(epsilons[j++]));  //TODO is this always safe?
                    } else {
                        gv.setEpsilon(epsilons[j++]);
                    }
                    setVertexEpsilon[gv.getVertexIndex()] = true;
                }
            }

            if (pair.getFirst() != null) {
                Gradient g = pair.getFirst();
                Map<String, INDArray> map = g.gradientForVariable();
                LinkedList<Triple<String, INDArray, Character>> tempList = new LinkedList<>();
                for (Map.Entry<String, INDArray> entry : map.entrySet()) {
                    String origName = entry.getKey();
                    String newName = current.getVertexName() + "_" + origName;
                    tempList.addFirst(new Triple<>(newName, entry.getValue(),
                            g.flatteningOrderForVariable(origName)));
                }
                for (Triple<String, INDArray, Character> t : tempList)
                    gradients.addFirst(t);
            }
        }

        //Now, add the gradients in the order we need them in for flattening (same as params order)
        Gradient gradient = new DefaultGradient(graph.getFlattenedGradients());
        INDArray totalGradient = null;
        Character order = null;
        for (Triple<String, INDArray, Character> t : gradients) {
            if(totalGradient == null) {
                totalGradient = t.getSecond().reshape(1, t.getSecond().length());
                order = t.getThird();
            } else {
                final INDArray grad = t.getSecond().reshape(1, t.getSecond().length());
                grad.setOrder(totalGradient.ordering());
                totalGradient = Nd4j.hstack(totalGradient, grad);
            }
        }
        gradient.setGradientFor(getVertexName(), totalGradient, order);

        return new Pair<>(gradient, outputEpsilons.toArray(new INDArray[0]));
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

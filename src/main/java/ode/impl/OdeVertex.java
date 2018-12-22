package ode.impl;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince54Integrator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
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
import org.nd4j.linalg.primitives.Pair;

/**
 * Implementation of an ODE block.
 *
 * @author Christian Skarby
 */
public class OdeVertex extends BaseGraphVertex {

    private final ComputationGraph graph;
    private final FirstOrderIntegrator odeSolver;

    public OdeVertex(ComputationGraph actualGraph,
                     String name,
                     int vertexIndex,
                     VertexIndices[] inputVertices,
                     VertexIndices[] outputVertices,
                     ComputationGraph innerGraph) {
        super(actualGraph, name, vertexIndex, inputVertices, outputVertices);
        this.graph = innerGraph;
        odeSolver = new DormandPrince54Integrator(1e-8, 10d, 1e-9, 1e-7);
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
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set");

        if(getInputs().length != 1) {
            throw new IllegalStateException("More than one input not supported!");
        }

        int dimcnt = 0;
        for (INDArray input : getInputs()) {
            dimcnt += input.length();
        }
        final int dim = dimcnt;

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
                    final INDArray input = fromDoubleVec(y, shape, workspaceMgr);
                    setInput(0, input, workspaceMgr);
                }
                first = false;
                final INDArray eval = Nd4j.toFlattened(evaluate(training, workspaceMgr));
                System.arraycopy(eval.toDoubleVector(), 0, yDot, 0, yDot.length);
            }
        };

        final double[] output = new double[dim];
        odeSolver.integrate(equation, 0, output, 1, output);

        return fromDoubleVec(output, shape, workspaceMgr);
    }

    private INDArray fromDoubleVec(double[] vec, long[] shape, LayerWorkspaceMgr workspaceMgr) {
        final INDArray input = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, 1, vec.length);
        for (int i = 0; i < vec.length; i++) {
            input.putScalar(i, vec[i]);
        }
        return input.reshape(shape);
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
                output = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, current.getInputs()[0]);
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

            current.clear();

        }
        return output;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        return null;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {

    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}

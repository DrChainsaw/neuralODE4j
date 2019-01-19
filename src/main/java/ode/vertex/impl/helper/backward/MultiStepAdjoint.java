package ode.vertex.impl.helper.backward;

import ode.solve.api.FirstOrderSolver;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * {@link OdeHelperBackward} using the adjoint method capable of handling multiple time steps. Gradients will be provided
 * for the last time step only.
 *
 * @author Christian Skarby
 */
public class MultiStepAdjoint implements OdeHelperBackward {

    private final FirstOrderSolver solver;
    private final INDArray time;
    private final int timeIndex;

    public MultiStepAdjoint(FirstOrderSolver solver, INDArray time, int timeIndex) {
        this.solver = solver;
        this.time = time;
        this.timeIndex = timeIndex;
        if(time.length() <= 2 || !time.isVector()) {
            throw new IllegalArgumentException("time must be a vector of size > 2! Was of shape: " + Arrays.toString(time.shape())+ "!");
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> solve(ComputationGraph graph, InputArrays input, MiscPar miscPars) {
        final INDArray zt = alignInShapeToTimeFirst(input.getLastOutput());
        final INDArray dL_dzt = alignInShapeToTimeFirst(input.getLossGradient());

        assertSizeVsTime(zt);
        assertSizeVsTime(dL_dzt);

        final INDArrayIndex[] timeIndexer = createIndexer(time);
        final INDArrayIndex[] ztIndexer = createIndexer(input.getLastOutput());
        final INDArrayIndex[] dL_dztIndexer= createIndexer(input.getLossGradient());

        Pair<Gradient, INDArray[]> gradients = null;
        final INDArray timeGradient = Nd4j.createUninitialized(time.shape());
        for (int step = 1; step < time.length(); step++) {
            final INDArray ztStep = getStep(ztIndexer,zt, step);
            final INDArray dL_dztStep = getStep(dL_dztIndexer, dL_dzt, step);

            final InputArrays stepInput = new InputArrays(
                    input.getLastInputs(),
                    ztStep,
                    dL_dztStep,
                    input.getRealGradientView()
            );
            timeIndexer[1] = NDArrayIndex.interval(step - 1, step+1);
            final OdeHelperBackward stepSolve = new SingleStepAdjoint(solver, time.get(timeIndexer), timeIndex);
            gradients = stepSolve.solve(graph, stepInput, miscPars);

            if(timeIndex != -1) {
                timeGradient.put(timeIndexer, gradients.getSecond()[timeIndex]);
            }
        }

        if(timeIndex != -1 && gradients != null) {
            gradients.getSecond()[timeIndex] = timeGradient;
        }

        return gradients;
    }

    private void assertSizeVsTime(INDArray array) {
        if(array.size(0) + 1 != time.length()) {
            throw new IllegalArgumentException("Must have one element less in first dimension compared to time!");
        }
    }

    private INDArrayIndex[] createIndexer(INDArray array) {
        final INDArrayIndex[] indexer = new INDArrayIndex[array.rank()];
        for(int dim = 0; dim < indexer.length; dim++) {
            indexer[dim] = NDArrayIndex.all();
        }
        return indexer;
    }

    private INDArray getStep(INDArrayIndex[] indexer, INDArray array, int step) {
        indexer[0] = NDArrayIndex.point(step-1);
        return array.get(indexer);
    }

    private INDArray alignInShapeToTimeFirst(INDArray array) {

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
}

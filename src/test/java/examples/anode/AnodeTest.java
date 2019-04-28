package examples.anode;

import ode.vertex.conf.ConcatZerosVertex;
import ode.vertex.conf.OdeVertex;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for neural ODE augmentation from https://arxiv.org/pdf/1904.01681.pdf
 *
 * @author Christian Skarby
 */
public class AnodeTest {

    /**
     * Test that implementation is complaint with claims of section 3 in paper part 1: Verify that the function
     * {g(1) = -1, g(-1) = 1} can not be learned by a normal neural ODE function
     */
    @Test
    public void failToSolveIntersectMapping() {
        final ComputationGraph graph = createGraph(false);
        assertEquals("Incorrect solution!", 0, graph.outputSingle(Nd4j.ones(1, 1)).getDouble(0), 1e-3);
        assertEquals("Incorrect solution!", 0, graph.outputSingle(Nd4j.ones(1, 1).negi()).getDouble(0), 1e-3);
    }

    /**
     * Test that implementation is complaint with claims of section 3 in paper part 1: Verify that the function
     * {g(1) = -1, g(-1) = 1} can be learned by an augmented neural ODE function
     */
    @Test
    public void augmentToSolveIntersectMapping() {
        final ComputationGraph graph = createGraph(true);
        assertEquals("Incorrect solution!", -1, graph.outputSingle(Nd4j.ones(1, 1)).getDouble(0), 1e-3);
        assertEquals("Incorrect solution!", 1, graph.outputSingle(Nd4j.ones(1, 1).negi()).getDouble(0), 1e-3);
    }

    @NotNull
    private ComputationGraph createGraph(boolean augment) {
        final NeuralNetConfiguration.Builder globalConf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(666)
                .updater(new Adam(0.05));

        final long nrofHidden = 8;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = globalConf
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.feedForward(1));

        String odeInput = "input";
        long odeSize = 1;
        if (augment) {
            graphBuilder.addVertex("augOde", new ConcatZerosVertex(1), "input");
            odeInput = "augOde";
            odeSize = 2;
        }

        graphBuilder

                .addVertex("ode", new OdeVertex.Builder(globalConf, "dense0",
                        new DenseLayer.Builder()
                                .nOut(nrofHidden)
                                .activation(new ActivationIdentity())
                                .build())
                        .addLayer("dense1",
                                new DenseLayer.Builder()
                                        .nOut(odeSize)
                                        .activation(new ActivationIdentity())
                                        .build(), "dense0").build(), odeInput);

        String output = "ode";
        if(augment) {
            graphBuilder.addVertex("subset", new SubsetVertex(0, 0), "ode");
            output = "subset";
        }

        final ComputationGraph graph = new ComputationGraph(graphBuilder
                .setOutputs("output")
                .addLayer("output",
                        new LossLayer.Builder()
                                .lossFunction(new LossMSE())
                                .activation(new ActivationIdentity())
                                .build(), output)
                .build()) {

            // Stupid ComputationGraph does not realize vertices may have parameters
            @Override
            public long numParams(boolean backwards) {
                long numParams = super.numParams(backwards);
                for (GraphVertex vertex : getVertices()) {
                    numParams += vertex.numParams();
                }
                return numParams;
            }
        };

        graph.init();

        final int batchSize = 4;
        final INDArray input = Nd4j.ones(1, 1);
        final DataSet ds = create1DDataSet(batchSize, input);
        for (int i = 0; i < 200; i++) {
            graph.fit(ds);
            //System.out.println(i + " [1 ,-1] => [" + graph.outputSingle(input) + ", " + graph.outputSingle(input.neg()) + "]");
        }
        return graph;
    }

    private DataSet create1DDataSet(int batchSize, INDArray inputProto) {
        final INDArray input = Nd4j.repeat(inputProto.reshape(inputProto.length()), batchSize);
        input.get(NDArrayIndex.interval(0, batchSize / 2), NDArrayIndex.all()).negi();
        return new DataSet(input, input.neg());
    }
}

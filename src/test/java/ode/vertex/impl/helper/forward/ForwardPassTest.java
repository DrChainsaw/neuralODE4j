package ode.vertex.impl.helper.forward;

import ode.vertex.impl.helper.NoTimeInput;
import ode.vertex.impl.helper.TimeInput;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ForwardPass}
 *
 * @author Christian Skarby
 */
public class ForwardPassTest {

    /**
     * Test that the derivative is a forward pass through the layers
     */
    @Test
    public void calculateDerivative() {
        final long nrofInputs = 5;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofInputs))
                .allowNoOutput(true)
                .addInputs("input")
                // Very simple dense layer which just performs element wise multiplication of input
                .addLayer("dense", new DenseLayer.Builder()
                        .nOut(nrofInputs)
                        .hasBias(false)
                        .weightInit(WeightInit.IDENTITY)
                        .activation(new ActivationIdentity())
                        .build(), "input")
                .build());
        graph.init();
        double mul = 1.23;
        graph.params().muli(mul);

        final INDArray input = Nd4j.arange(nrofInputs).reshape(1, nrofInputs);
        final INDArray expected = input.mul(mul);
        final INDArray actual = new ForwardPass(graph, LayerWorkspaceMgr.noWorkspaces(), false, new NoTimeInput(new INDArray[]{input}))
                .calculateDerivative(input, Nd4j.scalar(0), input.dup());

        assertEquals("Incorrect output!", expected, actual);
    }

    /**
     * Test that the derivative is a forward pass through the layers
     */
    @Test
    public void calculateDerivativeWithTimeDependency() {
        final long nrofInputs = 5;
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.feedForward(nrofInputs), InputType.feedForward(1))
                .allowNoOutput(true)
                .addInputs("input", "t")

                .addLayer("time", new DenseLayer.Builder()
                        .nOut(1)
                        .hasBias(false)
                        .weightInit(WeightInit.ONES)
                        .activation(new ActivationIdentity())
                        .build(), "t")
                .addLayer("denseOut", new DenseLayer.Builder()
                        .nOut(nrofInputs)
                        .hasBias(false)
                        .weightInit(WeightInit.ONES)
                        .activation(new ActivationIdentity())
                        .build(), "input", "time")
                .build());
        graph.init();

        final INDArray input = Nd4j.arange(nrofInputs).reshape(1, nrofInputs);
        final INDArray time = Nd4j.scalar(7.89);
        final INDArray expected = Nd4j.create(input.shape()).assign(input.sumNumber()).addi(time.getDouble(0));
        final INDArray actual = new ForwardPass(graph, LayerWorkspaceMgr.noWorkspaces(), false,
                new TimeInput(new INDArray[]{input}))
                .calculateDerivative(input, time, input.dup());

        assertArrayEquals("Incorrect output!", expected.toDoubleVector(), actual.toDoubleVector(), 1e-10);
    }
}
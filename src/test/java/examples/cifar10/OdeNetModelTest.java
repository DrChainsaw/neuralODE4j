package examples.cifar10;

import com.beust.jcommander.JCommander;
import ode.solve.conf.DummyIteration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link OdeNetModel}
 * 
 * @author Christian Skarby
 */
public class OdeNetModelTest {

    /**
     * Test that the model can be created and that it is possible to train two examples using only the A-block
     */
    @Test
    public void fitA() {
        final OdeNetModel factory = fit("-useABlock");
        assertEquals("Incorrect name!", "odenet_A", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using only the B-block
     */
    @Test
    public void fitB() {
        final OdeNetModel factory = fit("-useBBlock");
        assertEquals("Incorrect name!", "odenet_B", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using only the C-block
     */
    @Test
    public void fitC() {
        final OdeNetModel factory = fit("-useCBlock");
        assertEquals("Incorrect name!", "odenet_C", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using only the A- and
     * B-blocks
     */
    @Test
    public void fitAB() {
        final OdeNetModel factory = fit("-useABlock", "-useBBlock");
        assertEquals("Incorrect name!", "odenet_A_B", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using the A-, B- and
     * C-blocks
     */
    @Test
    public void fitABC() {
        final OdeNetModel factory = fit("-useABlock", "-useBBlock", "-useCBlock");
        assertEquals("Incorrect name!", "odenet_A_B_C", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using the A-, B- and
     * C-blocks as well as one initial reduction block
     */
    @Test
    public void fitABCReduce1() {
        final OdeNetModel factory = fit("-useABlock", "-useBBlock", "-useCBlock", "-nrofPreReductions", "1");
        assertEquals("Incorrect name!", "odenet_nrofPreReductions1_A_B_C", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using the A-, B- and
     * C-blocks as well as one initial reduction block and training of time steps
     */
    @Test
    public void fitABCReduce1TrainTime() {
        final OdeNetModel factory = fit("-useABlock", "-useBBlock", "-useCBlock", "-nrofPreReductions", "1", "-trainTime");
        assertEquals("Incorrect name!", "odenet_nrofPreReductions1_A_B_C_trainTime", factory.name());
    }

    private OdeNetModel fit(String ... args) {
        final OdeNetModel factory = createOdeNetModel(args);

        final ComputationGraph model = factory.create(new DummyIteration(3));
        model.fit(new MultiDataSet(
                new INDArray[] {Nd4j.randn(1, 3, 32, 32), Nd4j.arange(2).reshape(1,2)},
                new INDArray[] {Nd4j.randn(1,10)}));
        model.fit(new MultiDataSet(
                new INDArray[] {Nd4j.randn(1, 3, 32, 32), Nd4j.arange(2).reshape(1,2)},
                new INDArray[] {Nd4j.randn(1,10)}));
        return factory;
    }

    @NotNull
    private OdeNetModel createOdeNetModel(String ... args) {
        final OdeNetModel factory = new OdeNetModel();
        JCommander.newBuilder()
                .addObject(factory)
                .build()
                .parse(args);
        return factory;
    }
}
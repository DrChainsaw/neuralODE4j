package examples.cifar10;

import com.beust.jcommander.JCommander;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link InceptionResNetV1}
 *
 * @author Christian Skarby
 */
public class InceptionResNetV1Test {

    /**
     * Test that the model can be created and that it is possible to train two examples using only the A-block
     */
    @Test
    public void fitA() {
        final InceptionResNetV1 factory = fit("-nrofABlocks", "2", "-nrofBBlocks", "0", "-nrofCBlocks", "0");
        assertEquals("Incorrect name!", "inceptionResNetV1_A2", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using only the B-block
     */
    @Test
    public void fitB() {
        final InceptionResNetV1 factory = fit("-nrofABlocks", "0", "-nrofBBlocks", "2", "-nrofCBlocks", "0");
        assertEquals("Incorrect name!", "inceptionResNetV1_B2", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using only the C-block
     */
    @Test
    public void fitC() {
        final InceptionResNetV1 factory = fit("-nrofABlocks", "0", "-nrofBBlocks", "0", "-nrofCBlocks", "2");
        assertEquals("Incorrect name!", "inceptionResNetV1_C2", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using only the A- and
     * B-blocks
     */
    @Test
    public void fitAB() {
        final InceptionResNetV1 factory = fit("-nrofABlocks", "1", "-nrofBBlocks", "2", "-nrofCBlocks", "0");
        assertEquals("Incorrect name!", "inceptionResNetV1_A1_B2", factory.name());
    }

    /**
     * Test that the model can be created and that it is possible to train two examples using the A-, B- and
     * C-blocks
     */
    @Test
    public void fitABC() {
        final InceptionResNetV1 factory = fit("-nrofABlocks", "2", "-nrofBBlocks", "2", "-nrofCBlocks", "2");
        assertEquals("Incorrect name!", "inceptionResNetV1_A2_B2_C2", factory.name());
    }

    private InceptionResNetV1 fit(String ... args) {
        final InceptionResNetV1 factory = createInceptionResNetV1Model(args);

        final ComputationGraph model = factory.create();
        model.fit(new MultiDataSet(Nd4j.randn(new long[]{1, 3, 32, 32}), Nd4j.randn(1,10)));
        model.fit(new MultiDataSet(Nd4j.randn(new long[]{1, 3, 32, 32}), Nd4j.randn(1,10)));
        return factory;
    }

    @NotNull
    private InceptionResNetV1 createInceptionResNetV1Model(String ... args) {
        final InceptionResNetV1 factory = new InceptionResNetV1();
        JCommander.newBuilder()
                .addObject(factory)
                .build()
                .parse(args);
        return factory;
    }
}
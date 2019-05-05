package examples.anode;

import com.beust.jcommander.JCommander;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link OdeNetModelFactory} with augmentation
 */
public class AugOdeModelFactoryTest extends ModelFactoryTest {

    @Override
    protected ModelFactory factory() {
        final OdeNetModelFactory factory = new OdeNetModelFactory();
        JCommander.newBuilder()
                .addObject(factory)
                .build().parse("-nrofAugmentDims", "10");
        return factory;
    }

    /**
     * Test that the model really is augmented
     */
    @Test
    public void checkIsAugmented() {
        // Law of how many delimiters was it now again?
        assertTrue("No augmentation found!", factory().create(2).graph().getConfiguration().getTopologicalOrderStr().contains("aug"));
    }
}

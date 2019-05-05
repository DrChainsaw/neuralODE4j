package examples.anode;

/**
 * Test cases for {@link OdeNetModelFactory}
 *
 * @author Christian Skarby
 */
public class OdeNetModelFactoryTest extends ModelFactoryTest {

    @Override
    protected OdeNetModelFactory factory() {
        return new OdeNetModelFactory();
    }
}
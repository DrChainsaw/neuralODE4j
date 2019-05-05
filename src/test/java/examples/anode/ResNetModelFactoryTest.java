package examples.anode;

/**
 * Test cases for {@link ResNetModelFactory}
 *
 * @author Christian Skarby
 */
public class ResNetModelFactoryTest extends ModelFactoryTest{

    @Override
    protected ResNetModelFactory factory() {
        return new ResNetModelFactory();
    }
}
package examples.anode;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.jzy3d.colors.Color;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Generic test cases for implementations of {@link ModelFactory}
 *
 * @author Chrisian Skarby
 */
public abstract class ModelFactoryTest {

    /**
     * Create factory to test
     * @return
     */
    protected abstract ModelFactory factory();

    /**
     * Test that the model can be created and that it is possible to train for two examples
     */
    @Test
    public void fit1DInput() {
        final ModelFactory factory = factory();

        final ComputationGraph model = factory.create(1).graph();
        final INDArray parameters = model.params().dup();

        model.fit(new DataSet(Nd4j.ones(1,1), Nd4j.ones(1,1).negi()));
        model.fit(new DataSet(Nd4j.ones(1,1).negi(), Nd4j.ones(1,1)));

        assertNotEquals("Parameters not updated!", parameters, model.params());
    }

    /**
     * Test that the model can be created and that it is possible to train for two examples
     */
    @Test
    public void fit2DInput() {
        final ModelFactory factory = factory();

        final ComputationGraph model = factory.create(2).graph();
        final INDArray parameters = model.params().dup();

        model.fit(new DataSet(Nd4j.ones(1,2), Nd4j.ones(1,1).negi()));
        model.fit(new DataSet(Nd4j.ones(1,2).negi(), Nd4j.ones(1,1)));

        assertNotEquals("Parameters not updated!", parameters, model.params());
    }

    /**
     * Test plotting of 1D reference data
     */
    @Test
    public void plot1D() {
        final ModelFactory factory = factory();

        final StoreSeries3D series = new StoreSeries3D();
        factory.create(1).plotFeatures(new DataSet(Nd4j.ones(1,1), Nd4j.ones(1,1)), new NoPlot3D() {

            @Override
            public Series3D series(String label) {
                return series;
            }
        });
        assertEquals("Incorrect number of samples!", series.x.size(), series.y.size());
        assertEquals("Incorrect number of samples!", series.x.size(), series.z.size());
    }

    /**
     * Test plotting of 2D reference data
     */
    @Test
    public void plot2D() {
        final ModelFactory factory = factory();

        final StoreSeries3D series = new StoreSeries3D();
        factory.create(2).plotFeatures(new DataSet(Nd4j.ones(1,2), Nd4j.ones(1,1)), new NoPlot3D() {

            @Override
            public Series3D series(String label) {
                return series;
            }
        });
        assertEquals("Incorrect number of samples!", series.x.size(), series.y.size());
        assertEquals("Incorrect number of samples!", series.x.size(), series.z.size());
    }

    private static class StoreSeries3D implements Plot3D.Series3D {

        private final List<Double> x = new ArrayList<>();
        private final List<Double> y = new ArrayList<>();
        private final List<Double> z = new ArrayList<>();

        @Override
        public Plot3D.Series3D plot(List<Double> x, List<Double> y, List<Double> z) {
            this.x.addAll(x);
            this.y.addAll(y);
            this.z.addAll(z);
            return this;
        }

        @Override
        public Plot3D.Series3D color(Color color) {
            return this;
        }

        @Override
        public Plot3D.Series3D size(float size) {
            return this;
        }

        @Override
        public Plot3D.Series3D clear() {
            return this;
        }
    }
}
